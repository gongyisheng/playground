"""Fixed tensor buffers with lazy loading and LRU replacement."""

from collections import OrderedDict
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch


def _tensor_bindings(state: dict) -> Iterator[tuple[dict, Hashable, torch.Tensor]]:
    for key, value in state.items():
        if isinstance(value, dict):
            yield from _tensor_bindings(value)
        elif isinstance(value, torch.Tensor):
            yield state, key, value


@dataclass(slots=True)
class _Entry:
    key: Hashable
    state: dict
    bindings: list
    slot: int | None = None
    active: bool = False


class LRUMemoryPool:
    """Swap nested tensor dictionaries through a fixed number of reusable slots.

    Register CPU state, then use it inside ``with pool.acquire(key)``. The pool
    updates dictionary values in place, so consumers such as ``optimizer.state``
    need no movement logic. Each entry must have the same tensor shapes/dtypes in
    traversal order. Non-tensor metadata stays in its original dictionary.

    Use from one thread and one CUDA stream. Registered dictionaries and their
    tensors belong to the pool: do not change their structure, share them between
    entries, or retain/use device tensors outside a lease. CPU backing tensors
    are refreshed on eviction, not on every release.
    """

    def __init__(self, capacity: int, *, device: str = "cuda"):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are CPU and CUDA")
        self._stream = (
            torch.cuda.current_stream(self.device)
            if self.device.type == "cuda"
            else None
        )
        if self._stream is not None:
            self.device = self._stream.device
        self._cpu_buffer: dict[Hashable, _Entry] = {}
        # Empty slots come first; occupied slots follow in LRU order.
        self._gpu_buffer: OrderedDict[int, tuple[_Entry | None, list[torch.Tensor]]] = (
            OrderedDict()
        )

    def register(self, key: Hashable, state: dict) -> None:
        """Take ownership of CPU state without allocating any device buffers."""
        if key in self._cpu_buffer:
            raise KeyError(f"Already registered: {key!r}")
        if any(state is entry.state for entry in self._cpu_buffer.values()):
            raise ValueError("A state dictionary can only be registered once")
        bindings = list(_tensor_bindings(state))
        if not bindings:
            raise ValueError("Initialize state tensors on CPU before registering")
        if any(t.device.type != "cpu" or t.requires_grad for _, _, t in bindings):
            raise ValueError(
                "State tensors must be on CPU and must not require gradients"
            )
        layout = [(t.shape, t.dtype) for _, _, t in bindings]
        if self._gpu_buffer:
            _, template = next(iter(self._gpu_buffer.values()))
            expected = [(t.shape, t.dtype) for t in template]
        elif self._cpu_buffer:
            template = next(iter(self._cpu_buffer.values())).bindings
            expected = [(t.shape, t.dtype) for _, _, t in template]
        else:
            expected = layout
        if layout != expected:
            raise ValueError("All entries must have matching tensor shapes and dtypes")
        self._cpu_buffer[key] = _Entry(key, state, bindings)

    @contextmanager
    def acquire(self, key: Hashable) -> Iterator[dict]:
        """Load on demand, mark recently used, and protect the slot until release."""
        self._check_stream()
        entry = self._cpu_buffer[key]
        if entry.active:
            raise RuntimeError(f"Entry is already in use: {key!r}")
        self._onload(entry)
        entry.active = True
        try:
            yield entry.state
        finally:
            entry.active = False

    @torch.no_grad()
    def offload(self, key: Hashable) -> None:
        """Copy updates back to CPU and free the slot for reuse."""
        self._check_stream()
        entry = self._cpu_buffer[key]
        if entry.active:
            raise RuntimeError(f"Entry is still in use: {key!r}")
        if entry.slot is None:
            return
        slot = entry.slot
        _, tensors = self._gpu_buffer[slot]
        for tensor, (_, _, host) in zip(tensors, entry.bindings, strict=True):
            host.copy_(tensor, non_blocking=False)
        for container, name, host in entry.bindings:
            container[name] = host
        entry.slot = None
        self._gpu_buffer[slot] = None, tensors
        self._gpu_buffer.move_to_end(slot, last=False)

    def remove(self, key: Hashable) -> None:
        """Offload and unregister state; the caller keeps its CPU dictionary."""
        self.offload(key)
        del self._cpu_buffer[key]

    @property
    def resident_keys(self) -> tuple[Hashable, ...]:
        """Resident entries, from least to most recently used."""
        return tuple(
            entry.key for entry, _ in self._gpu_buffer.values() if entry is not None
        )

    @property
    def allocated_bytes(self) -> int:
        """Tensor storage allocated for all slots, excluding allocator overhead."""
        return sum(
            t.numel() * t.element_size()
            for _, tensors in self._gpu_buffer.values()
            for t in tensors
        )

    @torch.no_grad()
    def _onload(self, entry: _Entry) -> None:
        if entry.slot is not None:
            self._gpu_buffer.move_to_end(entry.slot)
            return
        if not self._gpu_buffer:
            buffers = [
                torch.empty(
                    (self.capacity, *host.shape), dtype=host.dtype, device=self.device
                )
                for _, _, host in entry.bindings
            ]
            self._gpu_buffer = OrderedDict(
                (slot, (None, [buffer[slot] for buffer in buffers]))
                for slot in range(self.capacity)
            )
        for slot, (owner, tensors) in self._gpu_buffer.items():
            if owner is None or not owner.active:
                if owner is not None:
                    self.offload(owner.key)
                break
        else:
            raise RuntimeError("All pool slots are in use")
        # Copy everything before rebinding, so a failed load leaves CPU state intact.
        for tensor, (_, _, host) in zip(tensors, entry.bindings, strict=True):
            tensor.copy_(host, non_blocking=False)
        for tensor, (container, name, _) in zip(tensors, entry.bindings, strict=True):
            container[name] = tensor
        entry.slot = slot
        self._gpu_buffer[slot] = entry, tensors
        self._gpu_buffer.move_to_end(slot)

    def _check_stream(self) -> None:
        if (
            self._stream is not None
            and torch.cuda.current_stream(self.device) != self._stream
        ):
            raise RuntimeError("Use the pool on the CUDA stream it was created on")
