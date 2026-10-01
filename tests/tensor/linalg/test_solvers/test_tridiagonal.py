import numpy as np
import pytest
import scipy

from pytensor import function
from pytensor import tensor as pt
from pytensor.tensor.linalg.solvers.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)
from tests import unittest_tools as utt


def tridiagonal_test_values(n, rng):
    """Diagonals of a tridiagonal matrix whose gttrf swaps rows at some steps, not all."""
    dl = rng.normal(size=n - 1) * np.where(np.arange(n - 1) % 2, 10.0, 0.1)
    d = rng.normal(size=n)
    du = rng.normal(size=n - 1)

    *_, ipiv, _ = scipy.linalg.lapack.dgttrf(dl, d, du)
    n_swaps = (ipiv[:-1] != np.arange(1, n)).sum()
    assert 0 < n_swaps < n - 1
    return dl, d, du


class TestTridiagonalLU(utt.InferShapeTester):
    @pytest.mark.parametrize("b_ndim", [1, 2])
    def test_infer_shape(self, b_ndim):
        rng = np.random.default_rng(utt.fetch_seed())
        dl, d, du = (pt.dvector(name) for name in ("dl", "d", "du"))
        b = pt.tensor("b", shape=(None,) * b_ndim, dtype="float64")
        factors = LUFactorTridiagonal()(dl, d, du)
        x = SolveLUFactorTridiagonal(b_ndim=b_ndim, transposed=False)(*factors, b)

        self._compile_and_check(
            [dl, d, du, b],
            [*factors, x],
            [*tridiagonal_test_values(6, rng), rng.normal(size=(6, 3)[:b_ndim])],
            (LUFactorTridiagonal, SolveLUFactorTridiagonal),
            warn=False,
        )

    @pytest.mark.parametrize(
        "static_input, n", [(0, 5), (1, 5), (2, 5), (1, 1), (1, 0)]
    )
    def test_static_shape(self, static_input, n):
        diagonals = [pt.vector(shape=(None,)) for _ in range(3)]
        diagonals[static_input] = pt.vector(shape=(n if static_input == 1 else n - 1,))
        dl, d, du, du2, ipiv = LUFactorTridiagonal()(*diagonals)

        assert d.type.shape == ipiv.type.shape == (n,)
        assert dl.type.shape == du.type.shape == (max(n - 1, 0),)
        assert du2.type.shape == (max(n - 2, 0),)

        x = SolveLUFactorTridiagonal(b_ndim=1, transposed=False)(
            dl, d, du, du2, ipiv, pt.vector()
        )
        assert x.type.shape == (n,)

    @pytest.mark.parametrize("transposed", [False, True])
    @pytest.mark.parametrize("b_ndim", [1, 2])
    def test_solve(self, b_ndim, transposed):
        dl, d, du = (pt.dvector(name) for name in ("dl", "d", "du"))
        b = pt.tensor("b", shape=(None,) * b_ndim, dtype="float64")
        x = SolveLUFactorTridiagonal(b_ndim=b_ndim, transposed=transposed)(
            *LUFactorTridiagonal()(dl, d, du), b
        )
        f = function([dl, d, du, b], x)

        rng = np.random.default_rng(utt.fetch_seed())
        for n in [0, 1, 2, 6]:
            dl_val, d_val, du_val = (
                rng.normal(size=size) for size in (max(n - 1, 0), n, max(n - 1, 0))
            )
            b_val = rng.normal(size=(n, 3)[:b_ndim])
            A_val = np.diag(d_val) + np.diag(dl_val, -1) + np.diag(du_val, 1)
            expected = scipy.linalg.solve(A_val.T if transposed else A_val, b_val)
            np.testing.assert_allclose(f(dl_val, d_val, du_val, b_val), expected)
