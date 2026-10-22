"""Fixed-shape PyTorch baseline; export on the configured Ascend build host."""
M, N, K = 256, 256, 256
DTYPE = "float16"


def make_inputs():
    import torch
    generator = torch.Generator().manual_seed(7)
    return (torch.randn(M, K, dtype=torch.float16, generator=generator),
            torch.randn(K, N, dtype=torch.float16, generator=generator))


def baseline(a, b):
    import torch
    return torch.matmul(a, b)
