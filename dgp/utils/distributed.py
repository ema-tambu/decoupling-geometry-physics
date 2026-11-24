import os
import torch
import torch.distributed as dist

def setup_distributed():
    """
    Initializes the distributed training environment.
    Supports both multi-GPU (DDP) and single-GPU/CPU fallback for local debugging.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        # NCCL is the highly optimized backend for NVIDIA GPUs
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    
    return distributed, rank, local_rank, world_size, device

def cleanup_distributed(distributed):
    """Safely tears down the distributed process group."""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

def distributed_average(total, count, device, world_size):
    """
    All-reduces (sums) a metric across all GPUs and computes the global average.
    """
    if world_size == 1:
        if isinstance(total, torch.Tensor):
            return (total / max(count, 1)).item()
        return total / max(count, 1)

    # Ensure total is a tensor on the correct device
    if not isinstance(total, torch.Tensor):
        total = torch.tensor(total, device=device, dtype=torch.float32)

    # Keep everything on the GPU. Stack the total and the count directly.
    count_tensor = torch.tensor(count, device=device, dtype=torch.float32)
    tensor_to_reduce = torch.stack([total, count_tensor])
    
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM)
    
    return (tensor_to_reduce[0] / tensor_to_reduce[1].clamp_min(1.0)).item()