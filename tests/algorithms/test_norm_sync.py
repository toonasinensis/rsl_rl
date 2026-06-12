"""Temporary check: distributed EmpiricalNormalization matches single-process stats."""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rsl_rl.modules import EmpiricalNormalization

DIM = 6
STEPS = 10
BATCH = 32
WORLD = 2


def make_data():
    g = torch.Generator().manual_seed(0)
    # data[step][rank]
    return [
        [torch.randn(BATCH, DIM, generator=g) * (r + 1) + r for r in range(WORLD)]
        for _ in range(STEPS)
    ]


def worker(rank: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)

    data = make_data()
    norm = EmpiricalNormalization(DIM)
    norm.train()
    for step in range(STEPS):
        norm.update(data[step][rank])

    # Reference: single process seeing the concatenated batches.
    ref = EmpiricalNormalization(DIM)
    ref.train()
    # Temporarily hide the process group so ref.update takes the local path.
    pg = dist.GroupMember.WORLD
    dist.GroupMember.WORLD = None
    for step in range(STEPS):
        ref.update(torch.cat(data[step], dim=0))
    dist.GroupMember.WORLD = pg

    mean_err = (norm._mean - ref._mean).abs().max().item()
    std_err = (norm._std - ref._std).abs().max().item()
    count_ok = norm.count.item() == ref.count.item() == STEPS * BATCH * WORLD
    print(f"[rank {rank}] mean_err={mean_err:.3e} std_err={std_err:.3e} count_ok={count_ok}")
    assert mean_err < 1e-5 and std_err < 1e-5 and count_ok, "MISMATCH"

    # Cross-rank consistency: buffers must be identical on all ranks.
    gathered = [torch.empty_like(norm._mean) for _ in range(WORLD)]
    dist.all_gather(gathered, norm._mean)
    assert torch.equal(gathered[0], gathered[1]), "ranks diverged"
    if rank == 0:
        print("OK: matches single-process reference and ranks are in sync")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=WORLD)
