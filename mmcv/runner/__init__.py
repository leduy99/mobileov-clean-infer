"""Minimal mmcv.runner shim for distributed info."""

def get_dist_info():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1

__all__ = ["get_dist_info"]
