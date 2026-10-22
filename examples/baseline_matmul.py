"""Set baseline={"kind":"python","path":"baseline_matmul.py"} in this folder's JSON."""
import torch


class MatMul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


def build_case(operator, seed):
    rng = torch.Generator().manual_seed(seed)
    dtype = getattr(torch, operator["dtype"])
    a = torch.randn(operator["m"], operator["k"], dtype=dtype, generator=rng)
    b = torch.randn(operator["k"], operator["n"], dtype=dtype, generator=rng)
    return MatMul(), (a, b)
