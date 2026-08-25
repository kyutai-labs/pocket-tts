import os

import torch
import torch.distributed as dist


def is_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _require_cuda() -> None:
    """Training on CPU is accidental (a mismatched torch build), not a use case."""
    if torch.cuda.is_available() or os.environ.get("POCKET_TTS_ALLOW_CPU") == "1":
        return
    if torch.version.cuda is None:
        hint = (
            "this is a CPU-only torch build: pyproject pins the CPU wheel index for "
            "inference, so installing the train extra replaces a CUDA torch. Reinstall "
            "torch afterwards, see training/README.md."
        )
    else:
        hint = (
            "torch is built against CUDA "
            f"{torch.version.cuda}; if the driver reports an older version, install a "
            "matching build, see training/README.md."
        )
    raise SystemExit(
        f"no CUDA device visible to torch (torch {torch.__version__}).\n{hint}\n"
        "Set POCKET_TTS_ALLOW_CPU=1 to run on CPU anyway."
    )


def init_distributed() -> torch.device:
    _require_cuda()
    if is_torchrun():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shutdown_distributed() -> None:
    """Hold every rank until all are done, then tear NCCL down.

    Without the barrier, multi-GPU training can crash at the end.
    """
    if not dist.is_initialized():
        return
    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()


def avg_across_ranks(value: float) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()
