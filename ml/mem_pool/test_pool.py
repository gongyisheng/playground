import pytest
import torch

from ml.mem_pool import LRUMemoryPool

DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_lru_reuses_slots_and_preserves_updates(device):
    pool = LRUMemoryPool(2, device=device)
    states = {
        key: {"nested": {"moment": torch.full((3,), float(i))}, "tag": key}
        for i, key in enumerate("abc")
    }
    hosts = {key: state["nested"]["moment"] for key, state in states.items()}
    for key, state in states.items():
        pool.register(key, state)
    assert pool.allocated_bytes == 0

    pointers = set()
    for key in "ab":
        with pool.acquire(key) as state:
            assert state is states[key]
            state["nested"]["moment"].add_(10)
            pointers.add(state["nested"]["moment"].data_ptr())
    assert pool.allocated_bytes == 2 * 3 * 4
    # A cache hit makes a most recently used, so c must evict b.
    with pool.acquire("a"):
        pass
    with pool.acquire("c") as state:
        assert state["nested"]["moment"].data_ptr() in pointers
        torch.testing.assert_close(
            state["nested"]["moment"].cpu(), torch.full((3,), 2.0)
        )
    assert pool.resident_keys == ("a", "c")
    assert states["b"]["nested"]["moment"] is hosts["b"]
    torch.testing.assert_close(hosts["b"], torch.full((3,), 11.0))

    with pool.acquire("b") as state:
        assert state["nested"]["moment"].data_ptr() in pointers
        torch.testing.assert_close(
            state["nested"]["moment"].cpu(), torch.full((3,), 11.0)
        )
        assert state["tag"] == "b"
    assert pool.allocated_bytes == 24


def test_active_entries_are_protected_and_exceptions_release_them():
    pool = LRUMemoryPool(2, device="cpu")
    for key in "abc":
        pool.register(key, {"value": torch.zeros(1)})
    with pytest.raises(LookupError, match="training failed"), pool.acquire("a") as a:
        a["value"].fill_(7)
        with pytest.raises(RuntimeError, match="already in use"), pool.acquire("a"):
            pass
        with pytest.raises(RuntimeError, match="still in use"):
            pool.offload("a")
        with (
            pool.acquire("b"),
            pytest.raises(RuntimeError, match="All pool slots"),
            pool.acquire("c"),
        ):
            pass
        # a is oldest, but its active lease forces replacement of b.
        with pool.acquire("c"):
            assert pool.resident_keys == ("a", "c")
            torch.testing.assert_close(a["value"], torch.tensor([7.0]))
        raise LookupError("training failed")
    pool.offload("a")
    with pool.acquire("a") as a:
        torch.testing.assert_close(a["value"], torch.tensor([7.0]))


@pytest.mark.parametrize(
    "invalid,allocated",
    [
        ({}, False),
        ({"value": torch.zeros(2)}, False),
        ({"value": torch.zeros(1, dtype=torch.float64)}, False),
        ({"value": torch.zeros(1, requires_grad=True)}, False),
        ({"value": torch.empty(1, device="meta")}, False),
        ({"value": torch.zeros(2)}, True),
        ({"value": torch.zeros(1, dtype=torch.float64)}, True),
    ],
)
def test_invalid_registration_does_not_change_the_pool(invalid, allocated):
    pool = LRUMemoryPool(1, device="cpu")
    original = {"value": torch.tensor([5.0])}
    pool.register("a", original)
    if allocated:
        with pool.acquire("a"):
            pass
        pool.remove("a")
    with pytest.raises(ValueError):
        pool.register("b", invalid)
    with pytest.raises(KeyError), pool.acquire("b"):
        pass
    if allocated:
        pool.register("a", original)
    with pool.acquire("a") as state:
        torch.testing.assert_close(state["value"], torch.tensor([5.0]))


def test_remove_restores_cpu_state_and_allows_slot_reuse():
    pool = LRUMemoryPool(2, device="cpu")
    host = torch.tensor([1.0])
    state = {"value": host}
    pool.register(None, state)
    pool.register("b", {"value": torch.tensor([9.0])})
    with pool.acquire(None):
        pointer = state["value"].data_ptr()
        state["value"].add_(4)
        with pytest.raises(RuntimeError, match="still in use"):
            pool.remove(None)
    with pool.acquire("b"):
        pass
    pool.remove(None)
    assert state["value"] is host
    torch.testing.assert_close(host, torch.tensor([5.0]))
    assert pool.resident_keys == ("b",)
    with pytest.raises(KeyError), pool.acquire(None):
        pass
    pool.register(None, state)
    with pool.acquire(None):
        assert state["value"].data_ptr() == pointer
    with pool.acquire("b") as other:
        torch.testing.assert_close(other["value"], torch.tensor([9.0]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "optimizer_type,amsgrad", [(torch.optim.Adam, False), (torch.optim.AdamW, True)]
)
def test_pooled_lora_training_matches_native_optimizer(device, optimizer_type, amsgrad):
    pool = LRUMemoryPool(1, device=device)
    optimizers, references = {}, {}
    for index, key in enumerate("ab"):
        params = [
            torch.nn.Parameter(
                torch.arange(8, device=device, dtype=torch.float64).reshape(2, 4) / 20
                + index
            ),
            torch.nn.Parameter(
                torch.arange(6, device=device, dtype=torch.float64).reshape(3, 2) / 20
            ),
        ]
        reference_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
        options = {
            "lr": 0.01,
            "amsgrad": amsgrad,
            "foreach": False,
            "capturable": device.startswith("cuda"),
        }
        optimizer = optimizer_type(params, **options)
        reference = optimizer_type(reference_params, **options)
        # Initialize host buffers once; native reference state remains lazy.
        for p in params:
            optimizer.state[p] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(p, device="cpu"),
                "exp_avg_sq": torch.zeros_like(p, device="cpu"),
            }
            if amsgrad:
                optimizer.state[p]["max_exp_avg_sq"] = torch.zeros_like(p, device="cpu")
        pool.register(key, optimizer.state)
        optimizers[key], references[key] = optimizer, reference

    x = torch.arange(20, device=device, dtype=torch.float64).reshape(5, 4) / 20
    base = torch.arange(12, device=device, dtype=torch.float64).reshape(3, 4) / 10
    for step, key in enumerate("abaabbab"):
        optimizer, reference = optimizers[key], references[key]
        for current in (optimizer, reference):
            a, b = current.param_groups[0]["params"]
            current.zero_grad(set_to_none=True)
            loss = (x @ base.T + x @ a.T @ b.T).square().mean()
            loss.backward()
            if step == 3:
                a.grad = None  # Per-parameter step counters must survive swapping.
        with pool.acquire(key):
            optimizer.step()
        reference.step()
        for actual, expected in zip(
            optimizer.param_groups[0]["params"],
            reference.param_groups[0]["params"],
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    for key in "ab":
        pool.offload(key)
        optimizer, reference = optimizers[key], references[key]
        for actual, expected in zip(
            optimizer.param_groups[0]["params"],
            reference.param_groups[0]["params"],
            strict=True,
        ):
            for name, tensor in optimizer.state[actual].items():
                assert tensor.device.type == "cpu"
                torch.testing.assert_close(
                    tensor, reference.state[expected][name].cpu(), rtol=0, atol=0
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_pool_rejects_a_different_stream():
    pool = LRUMemoryPool(1, device="cuda:0")
    state = {"value": torch.zeros(4)}
    pool.register("a", state)
    with pool.acquire("a"):
        state["value"].add_(3)
    with torch.cuda.stream(torch.cuda.Stream(device="cuda:0")):
        with pytest.raises(RuntimeError, match="CUDA stream"), pool.acquire("a"):
            pass
        with pytest.raises(RuntimeError, match="CUDA stream"):
            pool.offload("a")
    pool.offload("a")
    torch.testing.assert_close(state["value"], torch.full((4,), 3.0))
