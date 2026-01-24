import ctypes
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RustLibraryLoader:
    def __init__(self):
        self._lib = None
        self._load_lib()

    def _load_lib(self):
        search_paths = [
            Path(__file__).parent.parent
            / "training"
            / "rust_exts"
            / "audio_ds"
            / "target"
            / "release",
            Path("/usr/local/lib"),
        ]
        for path in search_paths:
            for name in ["libpocket_tts_audio_ds.so", "libpocket_tts_audio_ds.dylib"]:
                lib_path = path / name
                if lib_path.exists():
                    try:
                        self._lib = ctypes.CDLL(str(lib_path))
                        return
                    except Exception as exc:
                        logger.debug(
                            "Failed to load library from %s: %s", lib_path, exc
                        )
                        continue


_LOADER = RustLibraryLoader()
_LIB = _LOADER._lib
_AVAILABLE = _LIB is not None


def arange(start, stop, step=1.0):
    if _AVAILABLE:
        size = int((stop - start) / step)
        _LIB.arange.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        _LIB.arange.restype = ctypes.POINTER(ctypes.c_float)
        ptr = _LIB.arange(start, stop, step)
        return [ptr[i] for i in range(size)]
    else:
        import numpy as np

        return np.arange(start, stop, step)
