"""Common utility functions."""


def rank_zero_print(*args, **kwargs):
    """Print only on rank 0 (useful for distributed training)."""
    print(*args, **kwargs)
