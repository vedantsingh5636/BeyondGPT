import gc
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

_lifecycle_lock = threading.RLock()


@dataclass
class _ModelSlot:
    instance: object | None = None
    active_users: int = 0


_slots: dict[str, _ModelSlot] = {
    "embeddings": _ModelSlot(),
    "reranker": _ModelSlot(),
}


def get_or_load_model(
    name: str,
    factory: Callable[[], T],
    *,
    evict_other: str | None = None,
) -> T:
    """Return a cached model instance, evicting an idle sibling model before loading."""
    with _lifecycle_lock:
        slot = _slots[name]
        if slot.instance is not None:
            return slot.instance  # type: ignore[return-value]

        if evict_other is not None:
            other = _slots[evict_other]
            if other.instance is not None and other.active_users == 0:
                other.instance = None
                gc.collect()

        instance = factory()
        slot.instance = instance
        return instance


def clear_model(name: str, *, force: bool = False) -> bool:
    """Release a cached model if idle (or always when force=True)."""
    with _lifecycle_lock:
        slot = _slots[name]
        if not force and slot.active_users > 0:
            return False
        if slot.instance is not None:
            slot.instance = None
            gc.collect()
        return True


def get_model_state(name: str) -> tuple[bool, int]:
    """Return (is_loaded, active_users) for diagnostics and testing."""
    with _lifecycle_lock:
        slot = _slots[name]
        return slot.instance is not None, slot.active_users


@contextmanager
def model_lease(name: str) -> Iterator[None]:
    """Track active inference/upload usage so concurrent requests never evict an in-use model."""
    with _lifecycle_lock:
        _slots[name].active_users += 1
    try:
        yield
    finally:
        with _lifecycle_lock:
            _slots[name].active_users = max(0, _slots[name].active_users - 1)
