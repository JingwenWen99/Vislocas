import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """
    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
        # dist.all_reduce(tensor, op=ReduceOp.SUM)
    if average:
        world_size = dist.get_world_size()
        # for tensor in tensors:
        #     tensor.mul_(1.0 / world_size)
        # for tensor in tensors:
        #     tensor.float().mul_(1.0 / world_size)
        for index, tensor in enumerate(tensors):
            tensor = tensor / world_size
            tensors[index] = tensor

    return tensors


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_master_proc():
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() == 0
    else:
        return True