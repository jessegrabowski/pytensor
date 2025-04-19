import numpy as np

import pytensor.tensor as pt
from pytensor import config
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


def test_mlx_solve():
    rng = np.random.default_rng()

    A = pt.tensor("A", shape=(5, 5))
    b = pt.tensor("B", shape=(5, 5))

    out = pt.linalg.solve(A, b, b_ndim=2)

    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    b_val = rng.normal(size=(5, 5)).astype(config.floatX)

    compare_mlx_and_py(
        [A, b],
        [out],
        [A_val, b_val],
        # mlx complains about useless vmap (when there are no batch dims), so we need to include
        # local_remove_useless_blockwise rewrite for this test
        mlx_mode=mlx_mode.including("blockwise"),
    )


def test_mlx_SolveTriangular():
    rng = np.random.default_rng()

    A = pt.tensor("A", shape=(5, 5))
    b = pt.tensor("B", shape=(5, 5))

    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    b_val = rng.normal(size=(5, 5)).astype(config.floatX)

    out = pt.linalg.solve_triangular(
        A,
        b,
        trans=0,
        lower=True,
        unit_diagonal=False,
    )
    compare_mlx_and_py(
        [A, b], [out], [A_val, b_val], mlx_mode=mlx_mode.including("blockwise")
    )
