"""Fixed-shape Add baseline with a tail."""
NUMEL = 65537
DTYPE = "float16"


def make_inputs():
    import torch
    generator = torch.Generator().manual_seed(7)
    return tuple(torch.randn(NUMEL, dtype=torch.float16, generator=generator) for _ in range(2))


def baseline(a, b):
    return a + b
