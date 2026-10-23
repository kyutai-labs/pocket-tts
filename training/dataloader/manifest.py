"""Manifest access for large jsonl datasets."""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LazyEntries:
    """This rank's stride of manifest lines, read on demand via an offset index.

    Large manifests (tens of GB) would otherwise be held as per-process string
    lists by every loader subprocess. The index is a uint64 array of line-start
    offsets cached next to the manifest; the lines themselves stay on disk and
    are served through the (shared, reclaimable) page cache.
    """

    def __init__(self, path: str, rank: int, world_size: int) -> None:
        self._path = Path(path)
        self._rank = rank
        self._world = world_size
        self._offsets = None
        self._file = None
        self._ensure_index()

    def _index_path(self) -> Path:
        return self._path.with_name(self._path.name + ".idx")

    def _ensure_index(self) -> None:
        idx = self._index_path()
        src = self._path.stat()
        if idx.exists():
            meta = idx.stat()
            if meta.st_mtime >= src.st_mtime and meta.st_size > 8:
                return
        offsets = [0]
        with self._path.open("rb") as f:
            for line in f:
                offsets.append(offsets[-1] + len(line))
        arr = np.asarray(offsets, dtype=np.uint64)
        tmp = idx.with_name(idx.name + f".tmp.{os.getpid()}")
        arr.tofile(tmp)
        os.replace(tmp, idx)  # concurrent builders write identical content
        logger.info(f"indexed {len(arr) - 1} lines of {self._path.name}")

    def _load(self) -> None:
        all_offsets = np.fromfile(self._index_path(), dtype=np.uint64)
        starts, ends = all_offsets[:-1], all_offsets[1:]
        self._starts = starts[self._rank :: self._world]
        self._ends = ends[self._rank :: self._world]
        self._offsets = True

    def __len__(self) -> int:
        if self._offsets is None:
            self._load()
        return len(self._starts)

    def __getitem__(self, i: int) -> str:
        if self._offsets is None:
            self._load()
        if self._file is None:
            self._file = self._path.open("rb")
        self._file.seek(int(self._starts[i]))
        return self._file.read(int(self._ends[i] - self._starts[i])).decode()

    def __getstate__(self) -> dict[str, object]:
        return {"_path": self._path, "_rank": self._rank, "_world": self._world}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._offsets = None
        self._file = None


def load_entries(path: str, rank: int, world_size: int) -> LazyEntries:
    entries = LazyEntries(path, rank, world_size)
    logger.info(f"indexed {len(entries)} entries from {path} (rank {rank}/{world_size})")
    assert len(entries), f"no entries for rank {rank} in {path}"
    return entries
