from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, single_tag):
        if single_tag is None:
            return _TensorBackuperNormal(source_getter=source_getter)
        else:
            return _TensorBackuperNoop(source_getter=source_getter, single_tag=single_tag)

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()


class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag):
        super().__init__(source_getter=source_getter)
        self._single_tag = single_tag
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        ans = dict(self._source_getter())
        ans = {k: v.detach() for k, v in ans.items()}
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        return ans

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        torch.cuda.synchronize()

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        torch.cuda.synchronize()


def _compute_hash_dict(tensors: dict[str, torch.Tensor]):
    return {k: _compute_hash_tensor(v) for k, v in tensors.items()}


def _compute_hash_tensor(x: torch.Tensor):
    x = x.contiguous()
    x = x.view(-1)
    x = x.view(torch.uint32)
    x = x.sum()
    return x.item()


# --------------- helpers & tests ---------------

def _make_source_getter(params: dict[str, torch.Tensor]):
    def getter():
        return params.items()
    return getter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_normal():
    params = {
        "layer.weight": torch.randn(4, 4, device=DEVICE),
        "layer.bias": torch.randn(4, device=DEVICE),
    }
    backuper = TensorBackuper.create(source_getter=_make_source_getter(params), single_tag=None)

    # Backup original weights under two tags
    backuper.backup("actor")
    original_weight = params["layer.weight"].clone()
    original_bias = params["layer.bias"].clone()

    # Mutate (simulate training step)
    params["layer.weight"].add_(10.0)
    params["layer.bias"].add_(5.0)
    backuper.backup("ref")

    # Verify two tags exist
    assert backuper.backup_tags == ["actor", "ref"]

    # get() returns CPU tensors matching the snapshot
    assert torch.allclose(backuper.get("actor")["layer.weight"], original_weight.cpu())

    # Restore actor weights
    backuper.restore("actor")
    assert torch.allclose(params["layer.weight"], original_weight)
    assert torch.allclose(params["layer.bias"], original_bias)

    # Copy actor -> ref, verify ref now holds actor's data
    backuper.copy(src_tag="actor", dst_tag="ref")
    assert torch.allclose(backuper.get("ref")["layer.weight"], original_weight.cpu())


def test_noop():
    params = {"w": torch.tensor([1.0, 2.0], device=DEVICE)}
    backuper = TensorBackuper.create(source_getter=_make_source_getter(params), single_tag="actor")

    assert backuper.backup_tags == ["actor"]

    # Backup records hashes; restore passes when weights are unchanged
    backuper.backup("actor")
    backuper.restore("actor")

    # Mutating weights should make restore fail
    params["w"].fill_(999.0)
    try:
        backuper.restore("actor")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


if __name__ == "__main__":
    test_normal()
    print("PASSED: test_normal")
    test_noop()
    print("PASSED: test_noop")
    print("\nAll tests passed.")
