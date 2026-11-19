"""Reusable tensor slots with lazy onload/offload and LRU replacement."""

from .pool import LRUMemoryPool

__all__ = ["LRUMemoryPool"]
