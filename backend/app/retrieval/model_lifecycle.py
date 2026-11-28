import ctypes
import gc
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

import onnxruntime as ort
from fastembed.common.onnx_external_data import link_external_data

T = TypeVar("T")

_lifecycle_lock = threading.RLock()

_libc = None
if sys.platform.startswith("linux"):
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim.argtypes = [ctypes.c_size_t]
        _libc.malloc_trim.restype = ctypes.c_int
    except OSError:
        _libc = None


def trim_process_memory() -> bool:
    """Collect Python cycles and return freed glibc heap arenas to Linux when supported."""
    gc.collect()
    if _libc is None:
        return False
    try:
        return bool(_libc.malloc_trim(0))
    except (OSError, AttributeError):
        return False


def configure_low_memory_onnx_session(inner_model) -> None:
    """Load an ONNX Runtime CPU session tuned for minimal memory on 512 MB containers."""
    model_dir = inner_model._model_dir
    description = inner_model.model_description
    model_file = description.model_file
    additional_files = getattr(description, "additional_files", None) or []

    try:
        model_path = link_external_data(model_dir, model_file, additional_files)
    except OSError:
        model_path = model_dir / model_file

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    inner_model.model = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
        sess_options=session_options,
    )
    inner_model._ensure_tokenizer()


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
                trim_process_memory()

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
            trim_process_memory()
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
