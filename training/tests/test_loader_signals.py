import os
import signal
import time
from multiprocessing.synchronize import Event

import torch.multiprocessing as torch_mp

from training.dataloader.subproc import _ignore_stop_signals


def _child(ready: Event) -> None:
    _ignore_stop_signals()
    ready.set()
    time.sleep(60)


def test_loader_processes_survive_stop_signals() -> None:
    # scancel signals every process of the job; the loaders must outlive the trainer's
    # final step so it can checkpoint before exiting.
    ctx = torch_mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_child, args=(ready,), daemon=True)
    proc.start()
    try:
        assert ready.wait(timeout=60)
        pid = proc.pid
        assert pid is not None
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGUSR1)
        time.sleep(0.5)
        assert proc.is_alive()
    finally:
        proc.kill()
        proc.join(timeout=10)
    assert not proc.is_alive()
