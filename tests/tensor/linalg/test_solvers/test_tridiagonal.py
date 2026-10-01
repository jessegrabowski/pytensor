import numpy as np
import pytest
import scipy

import pytensor
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

    def test_lu_factor_grad(self):
        rng = np.random.default_rng(utt.fetch_seed())
        weights = [rng.normal(size=size) for size in (8, 9, 8, 7)]

        def cost(dl, d, du):
            factors = LUFactorTridiagonal()(dl, d, du)[:4]
            return sum(
                (w * factor).sum() for w, factor in zip(weights, factors, strict=True)
            )

        utt.verify_grad(cost, list(tridiagonal_test_values(9, rng)), rng=rng)

    @pytest.mark.parametrize("transposed", [False, True])
    @pytest.mark.parametrize("b_ndim", [1, 2])
    def test_solve_grad(self, b_ndim, transposed):
        rng = np.random.default_rng(utt.fetch_seed())
        dl, d, du, du2, ipiv, _ = scipy.linalg.lapack.dgttrf(
            *tridiagonal_test_values(9, rng)
        )
        b = rng.normal(size=(9, 3)[:b_ndim])

        def solve(dl, d, du, du2, b):
            return SolveLUFactorTridiagonal(b_ndim=b_ndim, transposed=transposed)(
                dl, d, du, du2, ipiv, b
            )

        utt.verify_grad(solve, [dl, d, du, du2, b], rng=rng)

    @pytest.mark.parametrize("transposed", [False, True])
    def test_grad_small_systems(self, transposed):
        dl, d, du = (pt.dvector(name) for name in ("dl", "d", "du"))
        b = pt.dmatrix("b")
        x = SolveLUFactorTridiagonal(b_ndim=2, transposed=transposed)(
            *LUFactorTridiagonal()(dl, d, du), b
        )
        f = function([dl, d, du, b], pytensor.grad(x.sum(), [d, b]))

        for n in [0, 1]:
            d_bar, b_bar = f(np.zeros(0), np.full(n, 2.0), np.zeros(0), np.ones((n, 3)))
            # A = 2 I, so x = b / 2 and each of the 3 columns contributes -1 / 2**2
            np.testing.assert_allclose(d_bar, np.full(n, -3 / 2**2))
            np.testing.assert_allclose(b_bar, np.full((n, 3), 1 / 2))
