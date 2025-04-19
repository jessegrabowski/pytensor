import warnings

import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.slinalg import Solve, SolveTriangular


@mlx_funcify.register(Solve)
def mlx_funcify_Solve(op, node, **kwargs):
    assume_a = op.assume_a

    if assume_a != "gen":
        warnings.warn(
            f"MLX solve does not support assume_a={op.assume_a}. Defaulting to assume_a='gen'.\n"
            f"If appropriate, you may want to set assume_a to one of 'sym', 'pos', 'her' or 'tridiagonal' to improve performance.",
            UserWarning,
        )

    def solve(a, b):
        # MLX only supports solve on CPU
        return mx.linalg.solve(a, b, stream=mx.cpu)

    return solve


@mlx_funcify.register(SolveTriangular)
def mlx_funcify_SolveTriangular(op, **kwargs):
    lower = op.lower

    def solve_triangular(A, b):
        return mx.linalg.solve_triangular(
            A,
            b,
            upper=not lower,
            stream=mx.cpu,  # MLX only supports solve_triangular on CPU
        )

    return solve_triangular
