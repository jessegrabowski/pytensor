from copy import copy
from itertools import product
from random import shuffle

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.compile.io import In
from pytensor.compile.mode import Mode
from pytensor.compile.sharedvalue import shared
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.utils import InconsistencyError
from pytensor.tensor import inplace
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blas import (
    BatchedDot,
    Dot22,
    Dot22Scalar,
    Gemm,
    Gemv,
    Ger,
    _batched_dot,
    _dot22,
    _dot22scalar,
    batched_dot,
    batched_tensordot,
    gemm,
    gemm_inplace,
    gemm_no_inplace,
    gemv,
    gemv_inplace,
    gemv_no_inplace,
    ger,
    ger_destructive,
)
from pytensor.tensor.math import Dot, dot, mean, mul, outer, sigmoid
from pytensor.tensor.rewriting.blas import local_dot22_to_dot22scalar, local_gemm_to_ger
from pytensor.tensor.type import (
    cmatrix,
    cscalar,
    dmatrix,
    drow,
    dscalar,
    fmatrix,
    fscalar,
    imatrix,
    int_dtypes,
    iscalar,
    ivector,
    matrices,
    matrix,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor4,
    vector,
    vectors,
    zmatrix,
    zscalar,
)
from tests import unittest_tools
from tests import unittest_tools as utt
from tests.tensor.utils import inplace_func, makeTester, random


if config.mode == "FAST_COMPILE":
    mode_not_fast_compile = "FAST_RUN"
else:
    mode_not_fast_compile = config.mode

mode_blas_opt = pytensor.compile.get_default_mode().including(
    "BlasOpt", "specialize", "InplaceBlasOpt"
)
mode_blas_opt = mode_blas_opt.excluding("c_blas")


def test_dot_eq():
    assert Dot() == Dot()


def sharedX(x, name):
    return shared(np.asarray(x, config.floatX), name=name)


class TestGemm:
    """
    This test suite is supposed to establish that gemm works as it is supposed to.
    """

    def setup_method(self):
        Gemm.debug = False

    @staticmethod
    def _gemm(z, a, x, y, b):
        assert a.shape == ()
        assert b.shape == ()
        return b * z + a * np.dot(x, y)

    def cmp(self, z_, a_, x_, y_, b_):
        for dtype in ["float32", "float64", "complex64", "complex128"]:
            z = np.asarray(z_, dtype=dtype)
            a = np.asarray(a_, dtype=dtype)
            x = np.asarray(x_, dtype=dtype)
            y = np.asarray(y_, dtype=dtype)
            b = np.asarray(b_, dtype=dtype)

            def cmp_linker(z, a, x, y, b, l):
                z, a, x, y, b = (np.asarray(p) for p in (z, a, x, y, b))
                z_orig = z.copy()
                tz, ta, tx, ty, tb = (
                    as_tensor_variable(p).type() for p in (z, a, x, y, b)
                )

                f = inplace_func(
                    [tz, ta, tx, ty, tb],
                    gemm_inplace(tz, ta, tx, ty, tb),
                    mode=Mode(optimizer=None, linker=l),
                )
                f(z, a, x, y, b)
                z_after = self._gemm(z_orig, a, x, y, b)

                # print z_orig, z_after, z, type(z_orig), type(z_after), type(z)
                unittest_tools.assert_allclose(z_after, z)
                if a == 0.0 and b == 1.0:
                    return
                elif z_orig.size == 0:
                    assert z.size == 0
                else:
                    assert np.any(z_orig != z)

            cmp_linker(copy(z), a, x, y, b, "c|py")
            cmp_linker(copy(z), a, x, y, b, "py")

            if not dtype.startswith("complex") and config.cxx:
                # If config.blas__ldflags is empty, PyTensor will use
                # a NumPy C implementation of [sd]gemm_.
                cmp_linker(copy(z), a, x, y, b, "c")

    def test_basic(self):
        Gemm.debug = True
        with pytest.raises(TypeError, match=Gemm.E_rank):
            gemm_no_inplace([1.0], 1.0, [1.0], [1.0], 1.0)

    def test_basic_1(self):
        with pytest.raises(TypeError, match=Gemm.E_rank):
            self.cmp(1.0, 0.0, 1.0, 1.0, 1.0)

    def test_basic_2(self):
        with pytest.raises(TypeError, match=Gemm.E_rank):
            self.cmp(2.0, 1.0, [3, 2, 1.0], [[1], [2], [3.0]], 1.0)

    def test_basic_4(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), 1.0, rng.random((3, 5)), rng.random((5, 4)), 0.0)

    def test_basic_5(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), 1.0, rng.random((3, 5)), rng.random((5, 4)), 1.0)

    def test_basic_6(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), 1.0, rng.random((3, 5)), rng.random((5, 4)), -1.0)

    def test_basic_7(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), 0.0, rng.random((3, 5)), rng.random((5, 4)), 0.0)

    def test_basic_8(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), 0.0, rng.random((3, 5)), rng.random((5, 4)), 0.6)

    def test_basic_9(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), 0.0, rng.random((3, 5)), rng.random((5, 4)), -1.0)

    def test_basic_10(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), -1.0, rng.random((3, 5)), rng.random((5, 4)), 0.0)

    def test_basic_11(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), -1.0, rng.random((3, 5)), rng.random((5, 4)), 1.0)

    def test_basic_12(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((3, 4)), -1.0, rng.random((3, 5)), rng.random((5, 4)), -1.0)

    def test_shape_0(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        self.cmp(rng.random((0, 4)), -1.0, rng.random((0, 5)), rng.random((5, 4)), -1.0)
        self.cmp(rng.random((3, 0)), -1.0, rng.random((3, 5)), rng.random((5, 0)), -1.0)
        self.cmp(rng.random((3, 4)), -1.0, rng.random((3, 0)), rng.random((0, 4)), -1.0)
        self.cmp(rng.random((0, 0)), -1.0, rng.random((0, 5)), rng.random((5, 0)), -1.0)
        self.cmp(rng.random((0, 0)), -1.0, rng.random((0, 0)), rng.random((0, 0)), -1.0)

    def test_factorised_scalar(self):
        a = matrix()
        b = matrix()
        s = shared(np.zeros((5, 5)).astype(config.floatX))

        lr1 = pt.constant(0.01).astype(config.floatX)
        lr2 = pt.constant(2).astype(config.floatX)
        l2_reg = pt.constant(0.0001).astype(config.floatX)

        # test constant merge with gemm
        f = function(
            [a, b],
            updates=[(s, lr1 * dot(a, b) + l2_reg * lr2 * s)],
            mode=mode_not_fast_compile,
        ).maker.fgraph.toposort()
        # [Gemm{inplace}(<TensorType(float64, matrix)>, 0.01,
        # <TensorType(float64, matrix)>, <TensorType(float64, matrix)>,
        # 2e-06)]
        assert len(f) == 1
        assert f[0].op == gemm_inplace

        # test factored scalar with merge
        f = function(
            [a, b],
            updates=[(s, lr1 * (dot(a, b) - l2_reg * s))],
            mode=mode_not_fast_compile,
        ).maker.fgraph.toposort()
        # [Gemm{inplace}(<TensorType(float64, matrix)>, 0.01,
        # <TensorType(float64, matrix)>, <TensorType(float64, matrix)>,
        # -2e-06)]
        assert len(f) == 1
        assert f[0].op == gemm_inplace

        # test factored scalar with merge and neg
        f = function(
            [a, b],
            updates=[(s, s - lr1 * (s * 0.0002 + dot(a, b)))],
            mode=mode_not_fast_compile,
        ).maker.fgraph.toposort()
        # [Gemm{inplace}(<TensorType(float64, matrix)>, -0.01,
        # <TensorType(float64, matrix)>, <TensorType(float64, matrix)>,
        # 0.999998)]
        assert len(f) == 1
        assert f[0].op == gemm_inplace

    def test_destroy_map0(self):
        # test that only first input can be overwritten.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        Z = as_tensor_variable(rng.random((2, 2)))
        with pytest.raises(InconsistencyError, match=Gemm.E_z_uniq):
            gemm_inplace(Z, 1.0, Z, Z, 1.0)

    def test_destroy_map1(self):
        # test that only first input can be overwritten.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        Z = as_tensor_variable(rng.random((2, 2)))
        A = as_tensor_variable(rng.random((2, 2)))
        with pytest.raises(InconsistencyError, match=Gemm.E_z_uniq):
            gemm_inplace(Z, 1.0, A, inplace.transpose_inplace(Z), 1.0)

    def test_destroy_map2(self):
        # test that only first input can be overwritten.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        Z = as_tensor_variable(rng.random((2, 2)))
        A = as_tensor_variable(rng.random((2, 2)))
        with pytest.raises(InconsistencyError, match=Gemm.E_z_uniq):
            gemm_inplace(Z, 1.0, inplace.transpose_inplace(Z), A, 1.0)

    def test_destroy_map3(self):
        # test that only first input can be overwritten
        rng = np.random.default_rng(seed=utt.fetch_seed())
        Z = as_tensor_variable(rng.random((2, 2)))
        A = as_tensor_variable(rng.random((2, 2)))
        with pytest.raises(InconsistencyError, match=Gemm.E_z_uniq):
            gemm_inplace(Z, 1.0, Z, A, 1.0)

    def test_destroy_map4(self):
        # test that dot args can be aliased
        rng = np.random.default_rng(seed=utt.fetch_seed())
        Z = shared(rng.random((2, 2)), name="Z")
        A = shared(rng.random((2, 2)), name="A")
        one = pt.constant(1.0).astype(Z.dtype)
        f = inplace_func([], gemm_inplace(Z, one, A, A, one))
        # TODO FIXME: This is a bad test
        f()
        f = inplace_func([], gemm_inplace(Z, one, A, A.T, one))
        # TODO FIXME: This is a bad test
        f()

    def test_transposes(self):
        # three square matrices which are not contiguous
        rng = np.random.default_rng(seed=utt.fetch_seed())
        A = rng.random((4, 5))[:, :4]
        B = rng.random((4, 5))[:, :4]
        C = rng.random((4, 5))[:, :4]

        def t(z, x, y, a=1.0, b=0.0, l="c|py", dt="float64"):
            z, a, x, y, b = (np.asarray(p, dtype=dt) for p in (z, a, x, y, b))
            # z_orig = z.copy()
            z_after = self._gemm(z, a, x, y, b)

            tz, ta, tx, ty, tb = (shared(p) for p in (z, a, x, y, b))

            # f = inplace_func([tz,ta,tx,ty,tb], gemm_inplace(tz,ta,tx,ty,tb),
            #                 mode = Mode(optimizer = None, linker=l))
            # f(z, a, x, y, b)
            f = inplace_func(
                [],
                gemm_inplace(tz, ta, tx, ty, tb),
                mode=Mode(optimizer=None, linker=l),
            )
            f()
            unittest_tools.assert_allclose(z_after, tz.get_value(borrow=True))
            f()
            unittest_tools.assert_allclose(z_after, tz.get_value(borrow=True))
            f()
            unittest_tools.assert_allclose(z_after, tz.get_value(borrow=True))

            # tz.value *= 0 # clear z's value
            y_T = ty.get_value(borrow=True).T
            ty.set_value(tx.get_value(borrow=True).T, borrow=True)
            tx.set_value(y_T, borrow=True)

            f()
            # test that the transposed version of multiplication gives
            # same answer
            unittest_tools.assert_allclose(z_after, tz.get_value(borrow=True).T)

        t(C, A, B)
        t(C.T, A, B)
        t(C, A.T, B, dt="float32")
        t(C, A, B.T)
        t(C.T, A.T, B)
        t(C, A.T, B.T, dt="float32")
        t(C.T, A, B.T)
        t(C.T, A.T, B.T, dt="float32")

        t(C, A[:, :2], B[:2, :])
        t(C.T, A[:, :2], B[:2, :], dt="float32")
        t(C, A[:2, :].T, B[:2, :])
        t(C.T, A[:2, :].T, B[:2, :], dt="float32")
        t(C, A[:2, :].T, B[:, :2].T)
        t(C.T, A[:2, :].T, B[:, :2].T)

        with pytest.raises(ValueError, match=r".*aligned.*"):
            t(C.T, A[:2, :], B[:, :2].T)

    def test_non_contiguous(self):
        # Like test_transposes but with matrices without any
        # continuous dimension
        rng = np.random.default_rng(seed=utt.fetch_seed())
        A = rng.random((4, 4, 3))
        B = rng.random((4, 4, 3))
        C = rng.random((4, 4, 3))

        def t(z, x, y, a=1.0, b=0.0, l="c|py", dt="float64"):
            z, a, x, y, b = (np.asarray(p, dtype=dt) for p in (z, a, x, y, b))
            z_orig = z.copy()
            z_after = np.zeros_like(z_orig)
            for i in range(3):
                z_after[:, :, i] = self._gemm(z[:, :, i], a, x[:, :, i], y[:, :, i], b)

            tz, ta, tx, ty, tb = (shared(p) for p in (z, a, x, y, b))
            for i in range(3):
                f_i = inplace_func(
                    [],
                    gemm_inplace(tz[:, :, i], ta, tx[:, :, i], ty[:, :, i], tb),
                    mode=Mode(optimizer=None, linker=l),
                )
                for j in range(3):
                    # tz will not _always_ be overwritten,
                    # and adding update={...} in the call to function()
                    # will create cycles, so we update by hand.
                    z_i = f_i()
                    z = tz.get_value(borrow=True, return_internal_type=True)
                    z[:, :, i] = z_i

                    unittest_tools.assert_allclose(
                        z_after[:, :, i], tz.get_value(borrow=True)[:, :, i]
                    )

                tz_i = gemm_no_inplace(tz[:, :, i], ta, tx[:, :, i], ty[:, :, i], tb)
                g_i = function(
                    [],
                    tz_i,
                    updates=[(tz, pt.set_subtensor(tz[:, :, i], tz_i))],
                    mode=Mode(optimizer=None, linker=l),
                )
                for j in range(3):
                    g_i()
                    unittest_tools.assert_allclose(
                        z_after[:, :, i], tz.get_value(borrow=True)[:, :, i]
                    )

        t(C, A, B)
        t(C.transpose((1, 0, 2)), A, B)
        t(C, A.transpose((1, 0, 2)), B, dt="float32")
        t(C, A, B.transpose((1, 0, 2)))
        t(C.transpose((1, 0, 2)), A.transpose((1, 0, 2)), B)
        t(C, A.transpose((1, 0, 2)), B.transpose((1, 0, 2)), dt="float32")
        t(C.transpose((1, 0, 2)), A, B.transpose((1, 0, 2)))
        t(
            C.transpose((1, 0, 2)),
            A.transpose((1, 0, 2)),
            B.transpose((1, 0, 2)),
            dt="float32",
        )


class TestGemmNoFlags:
    gemm = gemm_no_inplace
    M = 4
    N = 5
    K = 6
    slice_step = 3

    def get_variable(self, V, to_transpose, to_slice):
        if to_transpose:
            V = V.T
        if to_slice:
            V = V[:: self.slice_step]
        return V

    def get_function(
        self,
        dtype,
        transpose_A=False,
        transpose_B=False,
        transpose_C=False,
        slice_A=False,
        slice_B=False,
        slice_C=False,
    ):
        alpha = scalar(dtype=dtype, name="alpha")
        beta = scalar(dtype=dtype, name="beta")
        A = matrix(dtype=dtype, name="A")
        B = matrix(dtype=dtype, name="B")
        C = matrix(dtype=dtype, name="C")

        A1 = self.get_variable(A, transpose_A, slice_A)
        B1 = self.get_variable(B, transpose_B, slice_B)
        C1 = self.get_variable(C, transpose_C, slice_C)

        return function([alpha, A, B, beta, C], self.gemm(C1, alpha, A1, B1, beta))

    def generate_value(self, dtype, width, height, to_transpose, to_slice, rng):
        if to_slice:
            if to_transpose:
                shape = (height, width * self.slice_step)
            else:
                shape = (width * self.slice_step, height)
        else:
            if to_transpose:
                shape = (height, width)
            else:
                shape = (width, height)
        return rng.random(shape).astype(dtype)

    def get_data(
        self,
        rng,
        dtype,
        alpha,
        beta,
        transpose_A=False,
        transpose_B=False,
        transpose_C=False,
        slice_A=False,
        slice_B=False,
        slice_C=False,
    ):
        A = self.generate_value(dtype, self.M, self.N, transpose_A, slice_A, rng)
        B = self.generate_value(dtype, self.N, self.K, transpose_B, slice_B, rng)
        C = self.generate_value(dtype, self.M, self.K, transpose_C, slice_C, rng)
        return (alpha, A, B, beta, C)

    def get_value(self, V, to_transpose, to_slice):
        if to_transpose:
            V = V.T
        if to_slice:
            V = V[:: self.slice_step]
        return V

    def compute_ref(
        self,
        alpha,
        A,
        B,
        beta,
        C,
        transpose_A,
        transpose_B,
        transpose_C,
        slice_A,
        slice_B,
        slice_C,
    ):
        A = self.get_value(A, transpose_A, slice_A)
        B = self.get_value(B, transpose_B, slice_B)
        C = self.get_value(C, transpose_C, slice_C)
        return alpha * np.dot(A, B) + beta * C

    @config.change_flags(blas__ldflags="")
    def run_gemm(
        self,
        dtype,
        ALPHA,
        BETA,
        transpose_A,
        transpose_B,
        transpose_C,
        slice_A,
        slice_B,
        slice_C,
        rng,
    ):
        f = self.get_function(
            dtype, transpose_A, transpose_B, transpose_C, slice_A, slice_B, slice_C
        )
        values = self.get_data(
            rng,
            dtype,
            ALPHA,
            BETA,
            transpose_A,
            transpose_B,
            transpose_C,
            slice_A,
            slice_B,
            slice_C,
        )
        assert any(isinstance(node.op, Gemm) for node in f.maker.fgraph.apply_nodes)
        z_val = f(*values)
        assert z_val.dtype == dtype
        assert tuple(z_val.shape) == (self.M, self.K)
        ref_val = self.compute_ref(
            *(
                (
                    *values,
                    transpose_A,
                    transpose_B,
                    transpose_C,
                    slice_A,
                    slice_B,
                    slice_C,
                )
            )
        )
        unittest_tools.assert_allclose(ref_val, z_val)

    def test_gemm(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        dtypes = ("float32", "float64")
        scalars = (0, 1, -2)
        booleans = (False, True)
        # dtype, alpha, beta, transA, transB, transC, sliceA, sliceB, sliceC
        iterables = [dtypes] + ([scalars] * 2) + ([booleans] * 6)
        for dtype, alpha, beta, tA, tB, tC, sA, sB, sC in product(*iterables):
            self.run_gemm(dtype, alpha, beta, tA, tB, tC, sA, sB, sC, rng)


"""
This test suite ensures that Gemm is inserted where it belongs, and
that the resulting functions compute the same things as the originals.
"""


def XYZab():
    return matrix(), matrix(), matrix(), scalar(), scalar()


def just_gemm(i, o, ishapes=None, max_graphlen=0, expected_nb_gemm=1):
    if ishapes is None:
        ishapes = [(4, 3), (3, 5), (4, 5), (), ()]

    f = inplace_func(
        [In(ii, mutable=True, allow_downcast=True) for ii in i],
        o,
        mode="FAST_RUN",
        on_unused_input="ignore",
    )
    nb_gemm = 0
    for node in f.maker.fgraph.apply_nodes:
        assert not isinstance(
            node.op, Dot
        ), "_dot22 not changed to gemm_inplace in graph"
        assert node.op != _dot22
        if node.op == gemm_inplace:
            nb_gemm += 1
    assert nb_gemm == expected_nb_gemm, (nb_gemm, expected_nb_gemm)
    g = inplace_func(
        i,
        o,
        mode=Mode(linker="py", optimizer=None),
        allow_input_downcast=True,
        on_unused_input="ignore",
    )
    for node in g.maker.fgraph.apply_nodes:
        assert node.op != gemm_inplace, "gemm_inplace in original graph"

    graphlen = len(f.maker.fgraph.toposort())
    assert not (
        max_graphlen and (graphlen <= max_graphlen)
    ), f"graphlen={graphlen}>{max_graphlen}"

    rng = np.random.default_rng(unittest_tools.fetch_seed(234))
    r0 = f(*[np.asarray(rng.standard_normal(sh), config.floatX) for sh in ishapes])
    rng = np.random.default_rng(unittest_tools.fetch_seed(234))
    r1 = g(*[np.asarray(rng.standard_normal(sh), config.floatX) for sh in ishapes])
    max_abs_err = np.max(np.abs(r0[0] - r1[0]))

    # TODO: Needed to increase tolerance for certain tests when migrating to
    # Generators from RandomStates. Probably caused due to seed specified test.
    eps = 2.0e-8
    if config.floatX == "float32":
        eps = 2.0e-6
    assert max_abs_err <= eps, "GEMM is computing the wrong output. max_rel_err ="


@unittest_tools.assertFailure_fast
def test_gemm_opt0():
    # Many subgraphs whose dots can be eliminated
    X, Y, Z, a, b = XYZab()

    just_gemm([X, Y, Z, a, b], [dot(X, Y) * a + Z * b])
    just_gemm([X, Y, Z, a, b], [a * dot(X, Y) + b * Z])
    just_gemm([X, Y, Z, a, b], [b * Z + a * dot(X, Y)])
    just_gemm([X, Y, Z, a, b], [dot(X, Y) * a - Z * b])
    just_gemm([X, Y, Z, a, b], [a * dot(X, Y) - b * Z])
    just_gemm([X, Y, Z, a, b], [b * Z - a * dot(X, Y)])

    # with transposes (transposes should be pushed through dot in canonicalize)
    just_gemm([X, Y, Z, a, b], [b * Z.T - a * dot(Y.T, X.T)])
    just_gemm([X, Y, Z, a, b], [b * Z.T + a * b * dot(X, Y).T])
    just_gemm(
        [X, Y, Z, a, b],
        [b * Z + a * dot(X, Y).T],
        ishapes=[(5, 3), (3, 4), (4, 5), (), ()],
    )

    # with N multiplications instead of just one
    just_gemm([X, Y, Z, a, b], [(b * b) * Z * a + (a * a) * dot(X, Y) * b])
    just_gemm([X, Y, Z, a, b], [Z + dot(X, Y)])
    just_gemm([X, Y, Z, a, b], [Z * b + dot(X, Y)])
    just_gemm([X, Y, Z, a, b], [Z + a * b * a * dot(X, Y)])
    just_gemm([X, Y, Z, a, b], [(b * b) * Z * a - (a * a) * dot(X, Y) * b])
    just_gemm([X, Y, Z, a, b], [Z - dot(X, Y)])
    just_gemm([X, Y, Z, a, b], [Z * b - dot(X, Y)])
    just_gemm([X, Y, Z, a, b], [Z - a * b * a * dot(X, Y)])


@unittest_tools.assertFailure_fast
def test_gemm_opt_double_gemm():
    # This is the pattern that shows up in the autoencoder
    X, Y, Z, a, b = matrix(), matrix(), matrix(), scalar(), scalar()
    R, S, c = matrix(), matrix(), scalar()

    just_gemm(
        [X, Y, Z, a, b, R, S, c],
        [Z * c + a * dot(X, Y) + b * dot(R, S).T],
        ishapes=[(4, 3), (3, 5), (4, 5), (), (), (5, 9), (9, 4), ()],
        expected_nb_gemm=2,
    )

    ishapes = [(4, 3), (3, 5), (4, 5), (), (), (5, 9), (9, 4), ()]
    i = [X, Y, Z, a, b, R, S, c]
    o = [
        (
            a * dot(X, Y)
            + gemm_inplace(Z, b, S.T, R.T, pt.constant(1.0).astype(config.floatX))
        )
    ]
    f = inplace_func(
        [In(ii, mutable=True) for ii in i],
        o,
        mode="FAST_RUN",
        on_unused_input="ignore",
    )
    for node in f.maker.fgraph.apply_nodes:
        assert not isinstance(node.op, Dot)
        assert node.op != _dot22
    g = inplace_func(
        i,
        o,
        mode=Mode(linker="py", optimizer=None),
        on_unused_input="ignore",
    )

    rng = np.random.default_rng(unittest_tools.fetch_seed(234))
    r0 = f(*[np.asarray(rng.standard_normal(sh), config.floatX) for sh in ishapes])
    rng = np.random.default_rng(unittest_tools.fetch_seed(234))
    r1 = g(*[np.asarray(rng.standard_normal(sh), config.floatX) for sh in ishapes])
    max_abs_err = np.max(np.abs(r0[0] - r1[0]))

    # TODO: Needed to increase tolerance for certain tests when migrating to
    # Generators from RandomStates. Probably caused due to seed specified test.
    eps = 2.0e-8
    if config.floatX == "float32":
        eps = 2.0e-6
    assert max_abs_err <= eps, "GEMM is computing the wrong output. max_rel_err ="


def test_upcasting_scalar_nogemm():
    # Test that the optimization does not crash when the scale has an incorrect
    # dtype, and forces upcasting of the result
    v = fmatrix("v")
    w = fmatrix("w")
    t = fmatrix("t")
    alpha = dscalar("a")

    rval = dot(w, v) * alpha + t

    f = function([w, v, t, alpha], rval)
    t = f.maker.fgraph.toposort()
    assert sum(isinstance(n.op, Gemm) for n in t) == 0
    # pytensor.printing.debugprint(f, print_type=True)

    v = fmatrix("v")
    w = fmatrix("w")
    t = fmatrix("t")
    alpha = cscalar("a")

    with config.change_flags(on_opt_error="raise"):
        rval = dot(w, v) * alpha + t
        f = function([w, v, t, alpha], rval)

    t = f.maker.fgraph.toposort()
    assert sum(isinstance(n.op, Gemm) for n in t) == 0
    # pytensor.printing.debugprint(f, print_type=True)


def test_gemm_nested():
    X, Y, Z, a, b = (
        matrix("X"),
        matrix("Y"),
        matrix("Z"),
        scalar("a"),
        scalar("b"),
    )
    R, S, U, c, d = (
        matrix("R"),
        matrix("S"),
        matrix("U"),
        scalar("c"),
        scalar("d"),
    )

    just_gemm(
        [X, Y, Z, R, S, U, a, b, c, d],
        [a * Z - b * (c * dot(X, Y) + d * Z)],
        ishapes=[(2, 3), (3, 4), (2, 4), (2, 3), (3, 4), (2, 4), (), (), (), ()],
        max_graphlen=1,
    )
    # print "---------------------"
    just_gemm(
        [X, Y, Z, R, S, U, a, b, c, d],
        [a * Z - b * (c * dot(X, Y) + d * Z + c * Z)],
        ishapes=[(2, 3), (3, 4), (2, 4), (2, 3), (3, 4), (2, 4), (), (), (), ()],
        max_graphlen=1,
    )
    # print "---------------------"
    just_gemm(
        [X, Y, Z, R, S, U, a, b, c, d],
        [a * Z - b * (c * dot(X, Y) + d * Z + c * U)],
        ishapes=[(2, 3), (3, 4), (2, 4), (2, 3), (3, 4), (2, 4), (), (), (), ()],
        max_graphlen=3,
    )


def test_gemm_opt_wishlist():
    X, Y, Z, a, b = matrix(), matrix(), matrix(), scalar(), scalar()

    # with >2 additions of the same ``pt.dot(X, Y)`` term
    just_gemm([X, Y, Z, a, b], [(b * b) * Z * a + (a * a) * dot(X, Y) + b * dot(X, Y)])

    just_gemm([X, Y, Z, a, b], [Z + dot(X, Y) + dot(X, Y)])


def test_gemm_with_vector():
    # Many subgraphs whose dots can be eliminated.  This adds a
    # vector two the previous test, which triggers the long-sought GEMM
    # bug.

    X, Y, Z, a, b = XYZab()
    v = vector()

    def my_just_gemm(o):
        i = [X, Y, Z, a, b, v]
        ishapes = [(4, 3), (3, 5), (4, 5), (), (), (5,)]
        just_gemm(i, o, ishapes=ishapes)

    my_just_gemm([v + dot(X, Y) * a + Z * b])
    my_just_gemm([v + a * dot(X, Y) + b * Z])
    my_just_gemm([v + b * Z + a * dot(X, Y)])
    my_just_gemm([v + dot(X, Y) * a - Z * b])
    my_just_gemm([v + a * dot(X, Y) - b * Z])
    my_just_gemm([v + b * Z - a * dot(X, Y)])

    # with N multiplications instead of just one
    my_just_gemm([v + (b * b) * Z * a + (a * a) * dot(X, Y) * b])
    my_just_gemm([v + Z + dot(X, Y)])
    my_just_gemm([v + Z * b + dot(X, Y)])
    my_just_gemm([v + Z + a * b * a * dot(X, Y)])
    my_just_gemm([v + (b * b) * Z * a - (a * a) * dot(X, Y) * b])
    my_just_gemm([Z - dot(X, Y) + v])
    my_just_gemm([Z * b - dot(X, Y) + v])
    my_just_gemm([Z - a * b * a * dot(X, Y) + v])


def test_gemm_opt_vector_stuff():
    X, Y, a = matrix(), matrix(), scalar()
    u, v = vector(), vector()

    f = inplace_func([a, u, v], a + dot(u, v), mode="FAST_RUN")
    assert gemm_inplace not in [n.op for n in f.maker.fgraph.apply_nodes]

    f = inplace_func([a, u, X, Y], a * u + dot(X, Y), mode="FAST_RUN")
    assert gemm_inplace not in [n.op for n in f.maker.fgraph.apply_nodes]


def test_gemm_unrolled():
    # This test that the gemm optimizer remove the dot22 that was
    # present in the graph. Otherwise, this add a gemm, but still
    # compute the dot22.

    # This was not always the case in the with this the following code.

    batch_size = 100
    rep_size = 40
    rng = np.random.default_rng([1, 2, 3])

    for num_rounds in range(1, 10):
        W = sharedX(rng.standard_normal((rep_size, rep_size)), name="W")
        V = sharedX(np.zeros((batch_size, rep_size)), name="V")
        H = sharedX(np.zeros((batch_size, rep_size)), name="H")
        G = sharedX(np.zeros((batch_size, rep_size)), name="G")

        cur_V = V
        cur_H = H

        def update_V(cur_H):
            return sigmoid(dot(cur_H, W.T))

        def update_H(cur_V):
            return sigmoid(dot(cur_V, W) + dot(G, W.T))

        for i in range(num_rounds):
            cur_V = update_V(cur_H)
            cur_H = update_H(cur_V)

        unrolled_pytensor = function(
            [], updates=[(V, cur_V), (H, cur_H)], name="unrolled_pytensor"
        )
        nb_dot = sum(
            1
            for node in unrolled_pytensor.maker.fgraph.toposort()
            if isinstance(
                node.op,
                Dot | Dot22 | Gemm,
            )
        )
        # Each num_rounds add 3 dot, but one of them is always the same.
        # So the final graph should have 1 + 2* num_rounds dot variant op.
        assert nb_dot == num_rounds * 2 + 1, nb_dot

        # TODO FIXME: This is a bad test
        unrolled_pytensor()


def test_inplace0():
    # should fail to insert gemm_inplace because gemm_inplace would
    # create cycles
    X, Y, Z, a, b = (
        matrix("X"),
        matrix("Y"),
        matrix("Z"),
        scalar("a"),
        scalar("b"),
    )
    R, S, c = matrix("R"), matrix("S"), scalar("c")

    f = inplace_func([Z, b, R, S], [Z * (Z + b * dot(R, S).T)], mode="FAST_RUN")
    assert gemm_inplace not in [n.op for n in f.maker.fgraph.apply_nodes]
    assert gemm_no_inplace in [n.op for n in f.maker.fgraph.apply_nodes]

    # gemm_inplace should be inserted here, to work in-place on Z*c
    f = inplace_func(
        [X, Y, Z, a, b, R, S, c],
        [Z * (c * Z + a * dot(X, Y) + b * dot(R, S).T)],
        mode="FAST_RUN",
    )
    assert gemm_inplace in [n.op for n in f.maker.fgraph.apply_nodes]


def test_inplace1():
    X, Y, Z, a, b = XYZab()
    # with > 2 terms in the overall addition
    f = inplace_func([X, Y, Z], [Z + Z + dot(X, Y)], mode="FAST_RUN")
    # pytensor.printing.debugprint(f)
    # it doesn't work inplace because we didn't mark Z as mutable input
    assert [n.op for n in f.maker.fgraph.apply_nodes] == [gemm_no_inplace]


@pytest.mark.parametrize("linker", ("py", "cvm"))
@pytest.mark.parametrize("inplace", (False, True))
def test_gemm_broadcasting(inplace, linker):
    a, b = scalars("a", "b")
    z, x, y = matrices("z", "x", "y")

    mode = Mode(linker=linker)
    if inplace:
        out = gemm_inplace(z, a, x, y, b)
        f = pytensor.function([z, x, y, a, b], out, accept_inplace=True, mode=mode)
        assert [node.op for node in f.maker.fgraph.toposort()] == [gemm_inplace]
    else:
        out = gemm_no_inplace(z, a, x, y, b)
        f = pytensor.function([z, x, y, a, b], out, mode=mode)
        assert [node.op for node in f.maker.fgraph.toposort()] == [gemm_no_inplace]

    shapes_z = [(5, 3), (1, 3), (5, 1), (1, 1)]
    shapes_x = [(5, 4), (1, 4)]
    shapes_y = [(4, 3), (4, 1)]

    rng = np.random.default_rng()
    shuffle(shapes_z)
    shuffle(shapes_x)
    shuffle(shapes_y)
    for shape_z, shape_x, shape_y in product(shapes_z, shapes_x, shapes_y):
        z_v = rng.random(size=shape_z).astype(config.floatX)
        x_v = rng.random(size=shape_x).astype(config.floatX)
        y_v = rng.random(size=shape_y).astype(config.floatX)
        # We have to copy for the inplace case
        z_v_np = z_v.copy()
        np.testing.assert_allclose(
            f(z_v, x_v, y_v, 1, 1), z_v_np + np.dot(x_v, y_v), atol=2e-6
        )


def test_dot22():
    for dtype1 in ["float32", "float64", "complex64", "complex128"]:
        a = matrix(dtype=dtype1)
        for dtype2 in ["float32", "float64", "complex64", "complex128"]:
            b = matrix(dtype=dtype2)
            f = function([a, b], dot(a, b), mode=mode_blas_opt)
            topo = f.maker.fgraph.toposort()
            if dtype1 == dtype2:
                assert _dot22 in [x.op for x in topo], (dtype1, dtype2)
            else:
                check = [isinstance(x.op, Dot) for x in topo]
                assert any(check), (dtype1, dtype2)
            rng = np.random.default_rng(unittest_tools.fetch_seed())

            def cmp(a_shp, b_shp):
                av = rng.uniform(size=a_shp).astype(dtype1)
                bv = rng.uniform(size=b_shp).astype(dtype2)
                # TODO FIXME: This is a bad test
                f(av, bv)

            cmp((3, 4), (4, 5))
            cmp((0, 4), (4, 5))
            cmp((3, 0), (0, 5))
            cmp((3, 4), (4, 0))
            cmp((0, 4), (4, 0))
            cmp((0, 0), (0, 0))


def test_dot22scalar():
    # including does not seem to work for 'local_dot_to_dot22' and
    # 'local_dot22_to_dot22scalar'
    # TODO: exclude other optimizations in BlasOpt?
    # m = pytensor.compile.get_default_mode().including('local_dot_to_dot22',
    #                           'local_dot22_to_dot22scalar','specialize')
    # m = pytensor.compile.get_default_mode().including('BlasOpt', 'specialize')
    rng = np.random.default_rng(unittest_tools.fetch_seed())
    for dtype1 in ["complex64", "complex128"]:
        a = matrix("a", dtype=dtype1)
        for dtype2 in ["complex64", "complex128"]:
            b = matrix("b", dtype=dtype2)
            for dtype3 in ["complex64", "complex128"]:
                c = matrix("c", dtype=dtype3)
                for dtype4 in ["complex64", "complex128"]:
                    cst = pt.constant(0.2, dtype=dtype4)
                    cst2 = pt.constant(0.1, dtype=dtype4)

                    def check_dot22scalar(func, len_topo_scalar=-1):
                        topo = func.maker.fgraph.toposort()
                        ops = [x.op for x in topo]
                        dtype4_upcast = pytensor.scalar.upcast(dtype4, dtype1, dtype2)

                        if dtype1 == dtype2 == dtype3 == dtype4_upcast:
                            if len_topo_scalar > 0:
                                assert len(topo) == len_topo_scalar
                            assert _dot22scalar in ops, (dtype1, dtype2, dtype3, dtype4)
                        elif dtype1 == dtype2 == dtype4_upcast:
                            if not (len_topo_scalar > 0):
                                assert len(topo) == len_topo_scalar
                                assert _dot22scalar in ops, (
                                    dtype1,
                                    dtype2,
                                    dtype3,
                                    dtype4,
                                )
                            else:
                                # Currently there is a problem of
                                # optimization order The constant get
                                # upcasted to float64 before we try to
                                # merge it with the dot22 of
                                # float32. So this prevent the merge.
                                assert _dot22scalar in ops or _dot22 in ops, (
                                    dtype1,
                                    dtype2,
                                    dtype3,
                                    dtype4,
                                )

                        elif dtype1 == dtype2:
                            assert _dot22 in ops, (dtype1, dtype2, dtype3, dtype4)
                        else:
                            check = [isinstance(o, Dot) for o in ops]
                            assert any(check), (dtype1, dtype2, dtype3, dtype4)

                    def cmp(a_shp, b_shp, c_shp, sqr_shp=(5, 5)):
                        av = rng.uniform(size=a_shp).astype(dtype1)
                        bv = rng.uniform(size=b_shp).astype(dtype2)
                        cv = rng.uniform(size=c_shp).astype(dtype3)
                        sv = rng.uniform(size=sqr_shp).astype(dtype1)

                        if False:
                            f = function([a, b], cst * dot(a, b), mode=mode_blas_opt)
                            f.maker.fgraph.toposort()
                            check_dot22scalar(f, 1)

                            f(av, bv)

                        if True:
                            f = function(
                                [a, b, c], cst * c * dot(a, b), mode=mode_blas_opt
                            )
                            f.maker.fgraph.toposort()
                            check_dot22scalar(f, 2)

                            f(av, bv, cv)

                        f = function([a, b, c], c * cst * dot(a, b), mode=mode_blas_opt)
                        f.maker.fgraph.toposort()
                        check_dot22scalar(f, 2)
                        f(av, bv, cv)

                        # Here, canonicalize also seems needed
                        # TODO: add only the optimizations needed?
                        m2 = mode_blas_opt.including("canonicalize")
                        f = function([a, b, c], cst2 * c * cst * dot(a, b), mode=m2)
                        f.maker.fgraph.toposort()
                        check_dot22scalar(f, 2)
                        f(av, bv, cv)

                        if dtype1 == dtype2 == dtype3:
                            f = function([a, b, c], c * cst * a * dot(a, b), mode=m2)
                            f.maker.fgraph.toposort()
                            check_dot22scalar(f, 2)
                            f(sv, sv, sv)

                            f = function(
                                [a, b, c],
                                cst * c * a * dot(a, b),
                                mode=mode_blas_opt,
                            )
                            f.maker.fgraph.toposort()
                            # currently the canonizer don't always
                            # merge all Mul together...  dot22scalar
                            # optimizer does not do a recursive search
                            # therefore, it doesn't find potential
                            # matches of the scalar.  TODO: combine
                            # with the 'canonicalization' that is part
                            # of the Gemm optimizer.
                            #
                            #    assert _dot22scalar in [x.op for x in topo]
                            #    assert len(topo)==2
                            f(sv, sv, sv)

                            f = function([a, b, c], c * a * cst * dot(a, b), mode=m2)
                            f.maker.fgraph.toposort()
                            check_dot22scalar(f, 2)
                            f(sv, sv, sv)

                    cmp((3, 4), (4, 5), (3, 5))
                    cmp((0, 4), (4, 5), (0, 5))
                    cmp((3, 0), (0, 5), (3, 5))
                    cmp((3, 4), (4, 0), (3, 0), (0, 0))
                    cmp((0, 4), (4, 0), (0, 0))
                    cmp((0, 0), (0, 0), (0, 0))


def test_dot22scalar_cast():
    # Test that in `dot22_to_dot22scalar` we properly cast integers to floats.
    # Note that this test was failing before d5ff6904.
    A = dmatrix()
    for scalar_int_type in int_dtypes:
        y = scalar(dtype=scalar_int_type)
        f = function([A, y], dot(A, A) * y, mode=mode_blas_opt)
        assert _dot22scalar in [x.op for x in f.maker.fgraph.toposort()]
    A = fmatrix()
    for scalar_int_type in int_dtypes:
        y = scalar(dtype=scalar_int_type)
        f = function([A, y], dot(A, A) * y, mode=mode_blas_opt)
        if scalar_int_type in ["int32", "int64"]:
            assert _dot22 in [x.op for x in f.maker.fgraph.toposort()]
        else:
            assert _dot22scalar in [x.op for x in f.maker.fgraph.toposort()]


def test_local_dot22_to_dot22scalar():
    # This test that the bug in gh-1507 is really fixed
    A = dmatrix()
    mode = pytensor.compile.mode.get_default_mode()
    opt = in2out(local_dot22_to_dot22scalar)
    mode = mode.__class__(optimizer=opt)

    x = dscalar()
    y = dscalar()
    z = dscalar()
    # make sure to don't have dimshuffle as we don't opt those cases
    m = dmatrix()
    r = drow()
    for idx, node in enumerate(
        [
            # Old working cases
            mul(_dot22(A, A), x),
            mul(_dot22(A, A), x, y),
            mul(_dot22(A, A), x, r),
            mul(_dot22(A, A), m, x),
            mul(_dot22(A, A), x, m),
            mul(_dot22(A, A), x, (m * y)),
            mul(_dot22(A, A), (m * y), x),
            mul(_dot22(A, A), x, (r * y)),
            mul(_dot22(A, A), (r * y), x),
            mul(_dot22(A, A), (x * y), (m * x)),
            mul(_dot22(A, A), (r * y), (y * x)),
            # Case that was raising an assert that is fixed in gh-1507
            mul(_dot22(A, A), (m * y), m),
            mul(_dot22(A, A), m, (m * y)),
            mul(_dot22(A, A), (r * y), (m * x)),
            # assert fixed in gh-1507 and opt case added in gh-1515
            mul(_dot22(A, A), (m * y * z), m),
            mul(_dot22(A, A), m, (m * y * z)),
            # Opt case added in gh-1515
            mul(_dot22(A, A), mul(m, y, z), m),
            mul(_dot22(A, A), m, mul(m, y, z)),
            # Case that opt later in gh-1515
            mul(_dot22(A, A), (r * m), (m * x)),
        ]
    ):
        node2 = local_dot22_to_dot22scalar.transform(None, node.owner)
        assert node2
        f = function([x, y, z, m, r, A], node, mode=mode, on_unused_input="ignore")
        # TODO FIXME: This is a bad test
        f(0.1, 0.2, 0.3, [[1, 2], [3, 4]], [[5, 6]], [[7, 8], [9, 10]])


def test_dot_w_self():
    # This can trigger problems in the optimization because what would
    # normally be a gemm must not be because the output is aliased to
    # one of the inputs.

    A = shared(value=np.ones((2, 2)))
    B = matrix()

    p = dot(A, A) * B

    grad_res = grad(mean(p), A)
    f = function([B], p, updates=[(A, A - grad_res)])
    # tests correctness in debugmode
    # TODO FIXME: This is a bad test
    f(np.asarray([[0, 1], [2, 3]], dtype=config.floatX))


###############################################################################
# Tests for Gemv
###############################################################################


class TestGemv(unittest_tools.OptimizationTestMixin):
    def test_dot_vv(self):
        # Currently we generate a gemv for that case
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        v = shared(np.array(rng.uniform(size=(2,)), dtype="float32"))
        w = shared(np.array(rng.uniform(size=(2,)), dtype="float32"))
        f = function([], pytensor.tensor.dot(v, w), mode=mode_blas_opt)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, dot)
        self.assertFunctionContains1(f, Gemv(True))

        # Assert they produce the same output
        assert np.allclose(f(), np.dot(v.get_value(), w.get_value()))

    def test_dot_vm(self):
        # Test vector dot matrix
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        v = shared(np.array(rng.uniform(size=(2,)), dtype="float32"))
        m = shared(np.array(rng.uniform(size=(2, 3)), dtype="float32"))
        f = function([], pytensor.tensor.dot(v, m), mode=mode_blas_opt)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, dot)
        self.assertFunctionContains1(f, Gemv(True))

        # Assert they produce the same output
        assert np.allclose(f(), np.dot(v.get_value(), m.get_value()))
        # Assert it works when m has no contiguous dimension
        m.set_value(m.get_value(borrow=True)[::-1, ::-1], borrow=True)
        assert np.allclose(f(), np.dot(v.get_value(), m.get_value()))

    def test_dot_mv(self):
        # Test matrix dot vector
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        v = shared(np.array(rng.uniform(size=(2,)), dtype="float32"))
        m = shared(np.array(rng.uniform(size=(3, 2)), dtype="float32"))
        f = function([], pytensor.tensor.dot(m, v), mode=mode_blas_opt)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, dot)
        self.assertFunctionContains1(f, Gemv(True))

        # Assert they produce the same output
        assert np.allclose(f(), np.dot(m.get_value(), v.get_value()))
        # Assert it works when m has no contiguous dimension
        m.set_value(m.get_value(borrow=True)[::-1, ::-1], borrow=True)
        assert np.allclose(f(), np.dot(m.get_value(), v.get_value()))

    @staticmethod
    def t_gemv1(m_shp):
        # test vector2+dot(matrix,vector1)
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        v1 = shared(np.array(rng.uniform(size=(m_shp[1],)), dtype="float32"))
        v2_orig = np.array(rng.uniform(size=(m_shp[0],)), dtype="float32")
        v2 = shared(v2_orig)
        m = shared(np.array(rng.uniform(size=m_shp), dtype="float32"))

        f = function([], v2 + pytensor.tensor.dot(m, v1), mode=mode_blas_opt)

        # Assert they produce the same output
        assert np.allclose(f(), np.dot(m.get_value(), v1.get_value()) + v2_orig)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Gemv)
        assert topo[0].op.inplace is False

        # test the inplace version
        g = function(
            [], [], updates=[(v2, v2 + pytensor.tensor.dot(m, v1))], mode=mode_blas_opt
        )

        # Assert they produce the same output
        g()
        assert np.allclose(
            v2.get_value(), np.dot(m.get_value(), v1.get_value()) + v2_orig
        )
        topo = g.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Gemv)
        if config.mode != "FAST_COMPILE":
            assert topo[0].op.inplace is True

        # Do the same tests with a matrix with strides in both dimensions
        m.set_value(m.get_value(borrow=True)[::-1, ::-1], borrow=True)
        v2.set_value(v2_orig)
        assert np.allclose(f(), np.dot(m.get_value(), v1.get_value()) + v2_orig)
        g()
        assert np.allclose(
            v2.get_value(), np.dot(m.get_value(), v1.get_value()) + v2_orig
        )

    @pytest.mark.slow
    def test_gemv1(self):
        self.t_gemv1((3, 2))
        self.t_gemv1((0, 2))
        self.t_gemv1((3, 0))
        self.t_gemv1((0, 0))

    def test_gemv2(self):
        # test vector2+dot(vector1,matrix)
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        v1 = shared(np.array(rng.uniform(size=(2,)), dtype="float32"))
        v2_orig = np.array(rng.uniform(size=(3,)), dtype="float32")
        v2 = shared(v2_orig)
        m = shared(np.array(rng.uniform(size=(2, 3)), dtype="float32"))

        f = function([], v2 + pytensor.tensor.dot(v1, m), mode=mode_blas_opt)

        # Assert they produce the same output
        assert np.allclose(f(), np.dot(v1.get_value(), m.get_value()) + v2.get_value())
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(node.op, Gemv) for node in topo) == 1
        assert topo[-1].op.inplace is False

        # test the inplace version
        g = function(
            [], [], updates=[(v2, v2 + pytensor.tensor.dot(v1, m))], mode=mode_blas_opt
        )

        # Assert they produce the same output
        g()
        assert np.allclose(
            v2.get_value(), np.dot(v1.get_value(), m.get_value()) + v2_orig
        )
        topo = g.maker.fgraph.toposort()
        assert sum(isinstance(node.op, Gemv) for node in topo) == 1
        if config.mode != "FAST_COMPILE":
            assert topo[-1].op.inplace is True

        # Do the same tests with a matrix with strides in both dimensions
        m.set_value(m.get_value(borrow=True)[::-1, ::-1], borrow=True)
        v2.set_value(v2_orig)
        assert np.allclose(f(), np.dot(v1.get_value(), m.get_value()) + v2.get_value())
        g()
        assert np.allclose(
            v2.get_value(), np.dot(v1.get_value(), m.get_value()) + v2_orig
        )

    def test_gemv_broadcast(self):
        # test gemv with some broadcasted input
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        v1 = shared(np.array(rng.uniform(size=(2,)), dtype="float32"))
        v2_orig = np.array(rng.uniform(size=(1,)), dtype="float32")
        v2 = shared(v2_orig)
        m = shared(
            np.array(rng.uniform(size=(1, 2)), dtype="float32"),
            shape=(1, None),
        )
        o = pytensor.tensor.dot(m, v1)
        f = function([], o + v2, mode=mode_blas_opt)

        # Assert they produce the same output
        assert np.allclose(f(), np.dot(m.get_value(), v1.get_value()) + v2.get_value())
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(node.op, Gemv) for node in topo) == 1

        # call gemv directly for mixed broadcast pattern.
        o = gemv_no_inplace(v2, 0.5, m, v1, 0.25)
        f = function([], o, mode=mode_blas_opt)
        assert np.allclose(
            f(), 0.5 * np.dot(m.get_value(), v1.get_value()) + 0.25 * v2.get_value()
        )
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(node.op, Gemv) for node in topo) == 1

    def test_gemv_dimensions(self):
        A = matrix("A")
        x, y = vectors("x", "y")
        alpha = shared(np.asarray(1.0, dtype=config.floatX), name="alpha")
        beta = shared(np.asarray(1.0, dtype=config.floatX), name="beta")

        z = beta * y + alpha * dot(A, x)
        f = function([A, x, y], z)

        # Matrix value
        A_val = np.ones((5, 3), dtype=config.floatX)
        # Different vector length
        ones_3 = np.ones(3, dtype=config.floatX)
        ones_4 = np.ones(4, dtype=config.floatX)
        ones_5 = np.ones(5, dtype=config.floatX)
        ones_6 = np.ones(6, dtype=config.floatX)

        f(A_val, ones_3, ones_5)
        f(A_val[::-1, ::-1], ones_3, ones_5)
        with pytest.raises(ValueError):
            f(A_val, ones_4, ones_5)
        with pytest.raises(ValueError):
            f(A_val, ones_3, ones_6)
        with pytest.raises(ValueError):
            f(A_val, ones_4, ones_6)


# The following gemv tests were added in March 2011 by Ian Goodfellow
# and are based on the gemv tests from scipy
# http://projects.scipy.org/scipy/browser/trunk/scipy/linalg/tests/test_fblas.py?rev=6803
# NOTE: At the time these tests were written, pytensor did not have a
# conjugate function. If such a thing is ever added, the tests involving
# conjugate should be ported over as well.


def matrixmultiply(a, b):
    if len(b.shape) == 1:
        b_is_vector = True
        b = b[:, np.newaxis]
    else:
        b_is_vector = False
    assert a.shape[1] == b.shape[0]
    c = np.zeros((a.shape[0], b.shape[1]), np.common_type(a, b))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            s = 0
            for k in range(a.shape[1]):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    if b_is_vector:
        c = c.reshape((a.shape[0],))
    return c


class BaseGemv:
    mode = mode_blas_opt  # can be overridden with self.mode
    shared = staticmethod(shared)

    def get_data(self, x_stride=1, y_stride=1):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        mult = np.array(1, dtype=self.dtype)
        if self.dtype in [np.complex64, np.complex128]:
            mult = np.array(1 + 1j, dtype=self.dtype)
        alpha = np.array(1.0, dtype=self.dtype) * mult
        beta = np.array(1.0, dtype=self.dtype) * mult
        a = rng.standard_normal((3, 3)).astype(self.dtype) * mult
        x = np.arange(np.shape(a)[0] * x_stride, dtype=self.dtype) * mult
        y = np.arange(np.shape(a)[1] * y_stride, dtype=self.dtype) * mult
        return alpha, beta, a, x, y

    def test_simple(self):
        alpha, beta, a, x, y = (shared(value) for value in self.get_data())
        desired_oy = (
            alpha.get_value() * matrixmultiply(a.get_value(), x.get_value())
            + beta.get_value() * y.get_value()
        )

        oy = alpha * dot(a, x) + beta * y

        oy_func = function([], oy, mode=self.mode)

        oy_func.maker.fgraph.toposort()
        self.assertFunctionContains1(oy_func, self.gemv)

        oy_val = oy_func()

        assert_array_almost_equal(desired_oy, oy_val)

    def test_default_beta_y(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        a = shared(a_v)
        x = shared(x_v)

        desired_oy = matrixmultiply(a_v, x_v)

        oy = dot(a, x)

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv_inplace)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_simple_transpose(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = (shared(v) for v in vs)

        desired_oy = alpha_v * matrixmultiply(np.transpose(a_v), x_v) + beta_v * y_v

        oy = alpha * dot(a.T, x) + beta * y

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_x_stride(self):
        vs = self.get_data(x_stride=2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = (shared(v) for v in vs)

        desired_oy = alpha_v * matrixmultiply(a_v, x_v[::2]) + beta_v * y_v

        oy = alpha * dot(a, x[::2]) + beta * y

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_x_stride_transpose(self):
        vs = self.get_data(x_stride=2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = (shared(v) for v in vs)

        desired_oy = (
            alpha_v * matrixmultiply(np.transpose(a_v), x_v[::2]) + beta_v * y_v
        )

        oy = alpha * dot(a.T, x[::2]) + beta * y

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_y_stride(self):
        vs = self.get_data(y_stride=2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = (shared(v) for v in vs)

        desired_oy = alpha_v * matrixmultiply(a_v, x_v) + beta_v * y_v[::2]

        oy = alpha * dot(a, x) + beta * y[::2]

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_y_stride_transpose(self):
        vs = self.get_data(y_stride=2)
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = (shared(v) for v in vs)

        desired_oy = (
            alpha_v * matrixmultiply(np.transpose(a_v), x_v) + beta_v * y_v[::2]
        )

        oy = alpha * dot(a.T, x) + beta * y[::2]

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_a_strides(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = (shared(v) for v in vs)
        a_v = a_v[::-1, ::-1]
        a.set_value(
            a.get_value(borrow=True, return_internal_type=True)[::-1, ::-1], borrow=True
        )

        desired_oy = alpha_v * matrixmultiply(a_v, x_v) + beta_v * y_v

        oy = alpha * dot(a, x) + beta * y

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_a_strides_transpose(self):
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha, beta, a, x, y = (shared(v) for v in vs)
        a_v = a_v[::-1, ::-1]
        a.set_value(
            a.get_value(borrow=True, return_internal_type=True)[::-1, ::-1], borrow=True
        )

        desired_oy = alpha_v * matrixmultiply(np.transpose(a_v), x_v) + beta_v * y_v

        oy = alpha * dot(a.T, x) + beta * y

        oy_func = function([], oy, mode=self.mode)

        self.assertFunctionContains1(oy_func, self.gemv)

        oy_v = oy_func()
        assert_array_almost_equal(desired_oy, oy_v)

    def test_upcasting_scalar_nogemv(self):
        # Test that the optimization does not crash when the scale has
        # an incorrect dtype, and forces upcasting of the result
        # We put this test in this class to test it on the gpu too.
        vs = self.get_data()
        alpha_v, beta_v, a_v, x_v, y_v = vs
        alpha_v = alpha_v.astype("float64")
        a_v = a_v.astype("float32")
        x_v = x_v.astype("float32")
        y_v = y_v.astype("float32")

        alpha = dscalar("alpha")
        a = shared(a_v)
        x = shared(x_v)
        y = shared(y_v)

        rval = dot(a, x) * alpha + y

        f = function([alpha], rval, mode=self.mode)
        # this function is currently optimized so that the gemv is
        # done inplace on a temporarily allocated-buffer, which is
        # then scaled by alpha and to t with a fused elemwise.
        n_gemvs = 0
        # pytensor.printing.debugprint(f, print_type=True)
        for node in f.maker.fgraph.toposort():
            if node.op == self.gemv_inplace:
                n_gemvs += 1
                assert node.outputs[0].dtype == "float32"
        assert n_gemvs == 1, n_gemvs
        self.assertFunctionContains1(f, self.gemv_inplace)
        # TODO FIXME: This is a bad test
        f(alpha_v)


class TestSgemv(BaseGemv, unittest_tools.OptimizationTestMixin):
    dtype = np.float32
    gemv = gemv_no_inplace
    gemv_inplace = gemv_inplace


class TestDgemv(BaseGemv, unittest_tools.OptimizationTestMixin):
    dtype = np.float64
    gemv = gemv_no_inplace
    gemv_inplace = gemv_inplace


# The optimization to put Gemv don't work for complex type for now.
# See ticket 653.
# class TestCgemv(BaseGemv):
#    dtype = complex64

# class TestZgemv(BaseGemv):
#    dtype = complex128

###############################################################################
# Tests for Ger
###############################################################################


class TestGerMakeNode:
    def setup_method(self):
        self.iv = tensor(dtype="int32", shape=(None,))
        self.fv = tensor(dtype="float32", shape=(None,))
        self.fv1 = tensor(dtype="float32", shape=(1,))
        self.dv = tensor(dtype="float64", shape=(None,))
        self.dv1 = tensor(dtype="float64", shape=(1,))
        self.cv = tensor(dtype="complex64", shape=(None,))
        self.zv = tensor(dtype="complex128", shape=(None,))

        self.fv_2 = tensor(dtype="float32", shape=(None,))
        self.fv1_2 = tensor(dtype="float32", shape=(1,))
        self.dv_2 = tensor(dtype="float64", shape=(None,))
        self.dv1_2 = tensor(dtype="float64", shape=(1,))
        self.cv_2 = tensor(dtype="complex64", shape=(None,))
        self.zv_2 = tensor(dtype="complex128", shape=(None,))

        self.fm = fmatrix()
        self.dm = dmatrix()
        self.cm = cmatrix()
        self.zm = zmatrix()

        self.fa = fscalar()
        self.da = dscalar()
        self.ca = cscalar()
        self.za = zscalar()

    def test_works_on_all_valid_dtypes(self):
        assert self.fm.type == ger(self.fm, self.fa, self.fv, self.fv_2).type
        assert self.fm.type == ger(self.fm, self.fa, self.fv, self.fv_2).type
        assert self.fm.type == ger(self.fm, self.fa, self.fv, self.fv_2).type
        assert self.fm.type == ger(self.fm, self.fa, self.fv, self.fv_2).type

    def test_fails_on_invalid_dtypes(self):
        with pytest.raises(TypeError):
            ger(imatrix(), iscalar(), ivector(), ivector())

    def test_fails_for_nonscalar_alpha(self):
        with pytest.raises(TypeError):
            ger(self.fm, self.fm, self.fv, self.fv_2)
        # boundary case - fv1 has the right dtype and could be dimshuffled to a
        # scalar, but that's not make_node's job.
        with pytest.raises(TypeError):
            ger(self.fm, self.fv1, self.fv, self.fv_2)
        # actually doing the aforementioned dimshuffle makes it work
        assert (
            self.fm.type == ger(self.fm, self.fv1.dimshuffle(), self.fv, self.fv_2).type
        )

    def test_fails_for_nonmatrix_A(self):
        with pytest.raises(TypeError):
            ger(self.fv, self.fa, self.fv, self.fv_2)

    def test_fails_for_nonvector_x_or_y(self):
        with pytest.raises(TypeError):
            ger(self.fm, self.fa, self.fv.dimshuffle("x", 0), self.fv_2)
        with pytest.raises(TypeError):
            ger(self.fm, self.fa, self.fv, self.fv_2.dimshuffle("x", 0))

    def test_fails_for_mixed_dtypes(self):
        with pytest.raises(TypeError):
            ger(self.dm, self.fa, self.fv, self.fv_2)
        with pytest.raises(TypeError):
            ger(self.fm, self.da, self.fv, self.fv_2)
        with pytest.raises(TypeError):
            ger(self.fm, self.fa, self.dv, self.fv_2)
        with pytest.raises(TypeError):
            ger(self.fm, self.fa, self.fv, self.dv_2)
        with pytest.raises(TypeError):
            ger(self.cm, self.fa, self.fv, self.dv_2)
        with pytest.raises(TypeError):
            ger(self.cm, self.fa, self.fv, self.zv_2)


class TestGerOpContract(unittest_tools.OpContractTestMixin):
    def setup_method(self):
        self.ops = [ger, ger_destructive]

    def clone(self, op):
        return Ger(op.destructive)


class TestGer(unittest_tools.OptimizationTestMixin):
    shared = staticmethod(shared)

    def setup_method(self):
        self.mode = pytensor.compile.get_default_mode().including("fast_run")
        self.mode = self.mode.excluding("c_blas", "scipy_blas")
        dtype = self.dtype = "float64"  # optimization isn't dtype-dependent
        self.A = tensor(dtype=dtype, shape=(None, None))
        self.a = tensor(dtype=dtype, shape=())
        self.x = tensor(dtype=dtype, shape=(None,))
        self.y = tensor(dtype=dtype, shape=(None,))
        self.ger = ger
        self.ger_destructive = ger_destructive
        self.gemm = gemm_no_inplace

    def function(self, inputs, outputs, updates=None):
        if updates is None:
            updates = []
        return function(inputs, outputs, self.mode, updates=updates)

    def b(self, bval):
        return pt.as_tensor_variable(np.asarray(bval, dtype=self.dtype))

    def test_b_0_triggers_ger(self):
        # test local_gemm_to_ger opt
        assert local_gemm_to_ger.transform(
            None,
            gemm_no_inplace(
                self.A,
                self.a,
                self.x.dimshuffle(0, "x"),
                self.y.dimshuffle("x", 0),
                self.b(0),
            ).owner,
        )

    def test_b_1_triggers_ger(self):
        # test local_gemm_to_ger opt
        assert local_gemm_to_ger.transform(
            None,
            gemm_no_inplace(
                self.A,
                self.a,
                self.x.dimshuffle(0, "x"),
                self.y.dimshuffle("x", 0),
                self.b(1),
            ).owner,
        )

    def test_b_other_does_not_triggers_ger(self):
        # test local_gemm_to_ger opt
        assert not local_gemm_to_ger.transform(
            None,
            gemm_no_inplace(
                self.A,
                self.a,
                self.x.dimshuffle(0, "x"),
                self.y.dimshuffle("x", 0),
                self.b(1.5),
            ).owner,
        )

    def test_b_nonconst_does_not_triggers_ger(self):
        # test local_gemm_to_ger opt
        assert not local_gemm_to_ger.transform(
            None,
            gemm_no_inplace(
                self.A,
                self.a,
                self.x.dimshuffle(0, "x"),
                self.y.dimshuffle("x", 0),
                self.a,
            ).owner,
        )

    def test_outer(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        f = self.function([self.x, self.y], outer(self.x, self.y))
        self.assertFunctionContains(f, self.ger_destructive)
        f(
            rng.random(5).astype(self.dtype),
            rng.random(4).astype(self.dtype),
        ).shape == (5, 4)

    def test_A_plus_outer(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        f = self.function([self.A, self.x, self.y], self.A + outer(self.x, self.y))
        self.assertFunctionContains(f, self.ger)
        f(
            rng.random((5, 4)).astype(self.dtype),
            rng.random(5).astype(self.dtype),
            rng.random(4).astype(self.dtype),
        ).shape == (5, 4)
        f(
            rng.random((5, 4)).astype(self.dtype)[::-1, ::-1],
            rng.random(5).astype(self.dtype),
            rng.random(4).astype(self.dtype),
        ).shape == (5, 4)

    def test_A_plus_scaled_outer(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        f = self.function(
            [self.A, self.x, self.y], self.A + 0.1 * outer(self.x, self.y)
        )
        self.assertFunctionContains(f, self.ger)
        f(
            rng.random((5, 4)).astype(self.dtype),
            rng.random(5).astype(self.dtype),
            rng.random(4).astype(self.dtype),
        ).shape == (5, 4)
        f(
            rng.random((5, 4)).astype(self.dtype)[::-1, ::-1],
            rng.random(5).astype(self.dtype),
            rng.random(4).astype(self.dtype),
        ).shape == (5, 4)

    def test_scaled_A_plus_scaled_outer(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        f = self.function(
            [self.A, self.x, self.y],
            np.asarray(0.2, self.dtype) * self.A
            + np.asarray(0.1, self.dtype) * outer(self.x, self.y),
        )
        # Why gemm? This make the graph simpler did we test that it
        # make it faster?
        self.assertFunctionContains(f, self.gemm)
        f(
            rng.random((5, 4)).astype(self.dtype),
            rng.random(5).astype(self.dtype),
            rng.random(4).astype(self.dtype),
        ).shape == (5, 4)
        f(
            rng.random((5, 4)).astype(self.dtype)[::-1, ::-1],
            rng.random(5).astype(self.dtype),
            rng.random(4).astype(self.dtype),
        ).shape == (5, 4)

    def given_dtype(self, dtype, M, N, *, destructive=True):
        # test corner case shape and dtype
        rng = np.random.default_rng(unittest_tools.fetch_seed())

        A = tensor(dtype=dtype, shape=(None, None))
        x = tensor(dtype=dtype, shape=(None,))
        y = tensor(dtype=dtype, shape=(None,))

        f = self.function([A, x, y], A + 0.1 * outer(x, y))
        self.assertFunctionContains(
            f, self.ger_destructive if destructive else self.ger
        )
        f(
            rng.random((M, N)).astype(dtype),
            rng.random(M).astype(dtype),
            rng.random(N).astype(dtype),
        ).shape == (5, 4)
        f(
            rng.random((M, N)).astype(dtype)[::-1, ::-1],
            rng.random(M).astype(dtype),
            rng.random(N).astype(dtype),
        ).shape == (5, 4)

    def test_f32_0_0(self):
        return self.given_dtype("float32", 0, 0, destructive=config.floatX != "float32")

    def test_f32_1_0(self):
        return self.given_dtype("float32", 1, 0, destructive=config.floatX != "float32")

    def test_f32_0_1(self):
        return self.given_dtype("float32", 0, 1, destructive=config.floatX != "float32")

    def test_f32_1_1(self):
        return self.given_dtype("float32", 1, 1, destructive=config.floatX != "float32")

    def test_f32_4_4(self):
        return self.given_dtype("float32", 4, 4, destructive=config.floatX != "float32")

    def test_f32_7_1(self):
        return self.given_dtype("float32", 7, 1, destructive=config.floatX != "float32")

    def test_f32_1_2(self):
        return self.given_dtype("float32", 1, 2, destructive=config.floatX != "float32")

    def test_f64_4_5(self):
        return self.given_dtype("float64", 4, 5, destructive=False)

    def test_c64_7_1(self):
        return self.given_dtype("complex64", 7, 1)

    def test_c128_1_9(self):
        return self.given_dtype("complex128", 1, 9)

    def test_inplace(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        A = shared(rng.random((4, 5)).astype(self.dtype))
        f = self.function(
            [self.x, self.y],
            [],
            updates=[
                (A, A + pt.constant(0.1, dtype=self.dtype) * outer(self.x, self.y))
            ],
        )
        self.assertFunctionContains(f, self.ger_destructive)
        # TODO: Test something about the updated value of `A`
        f(
            rng.random(4).astype(self.dtype),
            rng.random(5).astype(self.dtype),
        )

        A.set_value(
            A.get_value(borrow=True, return_internal_type=True)[::-1, ::-1], borrow=True
        )
        f(
            rng.random(4).astype(self.dtype),
            rng.random(5).astype(self.dtype),
        )


class TestBlasStrides:
    dtype = "float64"
    mode = pytensor.compile.get_default_mode()
    mode = mode.including("fast_run").excluding("gpu", "c_blas", "scipy_blas")

    def random(self, *shape, rng=None):
        return np.asarray(rng.random(shape), dtype=self.dtype)

    def cmp_dot22(self, b_shp, c_shp, rng):
        av = np.zeros((0, 0), dtype=self.dtype)
        bv = self.random(*b_shp, rng=rng)
        cv = self.random(*c_shp, rng=rng)

        a = shared(av, "a")
        b = shared(bv, "b")
        c = shared(cv, "c")

        b_t = shared(bv.T, "b.T")
        c_t = shared(cv.T, "c.T")

        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)
        bt_dev = b_t.get_value(borrow=False, return_internal_type=True)
        ct_dev = c_t.get_value(borrow=False, return_internal_type=True)

        f_nn = function([], [], updates=[(a, dot(b, c))], mode=self.mode)
        # print 'class name:', self.__class__.__name__
        # pytensor.printing.debugprint(f_nn)
        f_nt = function([], [], updates=[(a, dot(b, c_t.T))], mode=self.mode)
        f_tn = function([], [], updates=[(a, dot(b_t.T, c))], mode=self.mode)
        f_tt = function([], [], updates=[(a, dot(b_t.T, c_t.T))], mode=self.mode)

        # Try with all stride patterns, and all transposed pattern
        for step_signs in product((-1, 1), repeat=4):
            for step in (1, 2):
                b_step1, b_step2, c_step1, c_step2 = (s * step for s in step_signs)

                b.set_value(b_dev.copy()[::b_step1, ::b_step2], borrow=True)
                c.set_value(c_dev.copy()[::c_step1, ::c_step2], borrow=True)
                b_t.set_value(bt_dev.copy()[::b_step2, ::b_step1], borrow=True)
                c_t.set_value(ct_dev.copy()[::c_step2, ::c_step1], borrow=True)

                # Numpy result
                a_n = np.dot(bv[::b_step1, ::b_step2], cv[::c_step1, ::c_step2])

                f_nn()
                assert np.allclose(a.get_value(), a_n)

                f_nt()
                assert np.allclose(a.get_value(), a_n)

                f_tn()
                assert np.allclose(a.get_value(), a_n)

                f_tt()
                assert np.allclose(a.get_value(), a_n)

    def test_dot22(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        self.cmp_dot22((3, 4), (4, 5), rng)
        self.cmp_dot22((1, 4), (4, 5), rng)
        self.cmp_dot22((3, 4), (4, 1), rng)
        self.cmp_dot22((3, 1), (1, 1), rng)
        self.cmp_dot22((1, 4), (4, 1), rng)
        self.cmp_dot22((3, 1), (1, 5), rng)
        self.cmp_dot22((0, 4), (4, 5), rng)
        self.cmp_dot22((0, 4), (4, 1), rng)
        self.cmp_dot22((0, 1), (1, 5), rng)
        self.cmp_dot22((3, 4), (4, 0), rng)
        self.cmp_dot22((3, 0), (0, 5), rng)
        self.cmp_dot22((0, 4), (4, 0), rng)
        self.cmp_dot22((0, 0), (0, 0), rng)

    def cmp_dot22scalar(self, b_shp, c_shp, rng):
        av = np.zeros((0, 0), dtype=self.dtype)
        bv = self.random(*b_shp, rng=rng)
        cv = self.random(*c_shp, rng=rng)
        l = np.float32(0.2)

        a = shared(av, "a")
        b = shared(bv, "b")
        c = shared(cv, "c")

        b_t = shared(bv.T, "b.T")
        c_t = shared(cv.T, "c.T")

        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)
        bt_dev = b_t.get_value(borrow=False, return_internal_type=True)
        ct_dev = c_t.get_value(borrow=False, return_internal_type=True)

        f_nn = function([], [], updates=[(a, l * dot(b, c))], mode=self.mode)
        f_nt = function([], [], updates=[(a, l * dot(b, c_t.T))], mode=self.mode)
        f_tn = function([], [], updates=[(a, l * dot(b_t.T, c))], mode=self.mode)
        f_tt = function([], [], updates=[(a, l * dot(b_t.T, c_t.T))], mode=self.mode)

        # Try with all stride patterns, and all transposed pattern
        for step_signs in product((-1, 1), repeat=4):
            for step in (1, 2):
                b_step1, b_step2, c_step1, c_step2 = (s * step for s in step_signs)

                b.set_value(b_dev.copy()[::b_step1, ::b_step2], borrow=True)
                c.set_value(c_dev.copy()[::c_step1, ::c_step2], borrow=True)
                b_t.set_value(bt_dev.copy()[::b_step2, ::b_step1], borrow=True)
                c_t.set_value(ct_dev.copy()[::c_step2, ::c_step1], borrow=True)

                # Numpy result
                a_n = l * np.dot(bv[::b_step1, ::b_step2], cv[::c_step1, ::c_step2])

                f_nn()
                assert np.allclose(a.get_value(), a_n)

                f_nt()
                assert np.allclose(a.get_value(), a_n)

                f_tn()
                assert np.allclose(a.get_value(), a_n)

                f_tt()
                assert np.allclose(a.get_value(), a_n)

    def test_dot22scalar(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        self.cmp_dot22scalar((3, 4), (4, 5), rng)
        self.cmp_dot22scalar((1, 4), (4, 5), rng)
        self.cmp_dot22scalar((3, 4), (4, 1), rng)
        self.cmp_dot22scalar((3, 1), (1, 1), rng)
        self.cmp_dot22scalar((1, 4), (4, 1), rng)
        self.cmp_dot22scalar((3, 1), (1, 5), rng)
        self.cmp_dot22scalar((0, 4), (4, 5), rng)
        self.cmp_dot22scalar((0, 4), (4, 1), rng)
        self.cmp_dot22scalar((0, 1), (1, 5), rng)
        self.cmp_dot22scalar((3, 4), (4, 0), rng)
        self.cmp_dot22scalar((3, 0), (0, 5), rng)
        self.cmp_dot22scalar((0, 4), (4, 0), rng)
        self.cmp_dot22scalar((0, 0), (0, 0), rng)

    def cmp_gemm(self, a_shp, b_shp, c_shp, rng):
        av = self.random(*a_shp, rng=rng)
        bv = self.random(*b_shp, rng=rng)
        cv = self.random(*c_shp, rng=rng)
        l = np.float32(0.2)

        a = shared(av, "a")
        b = shared(bv, "b")
        c = shared(cv, "c")

        a_t = shared(av.T, "a.T")
        b_t = shared(bv.T, "b.T")
        c_t = shared(cv.T, "c.T")

        a_dev = a.get_value(borrow=False, return_internal_type=True)
        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)
        bt_dev = b_t.get_value(borrow=False, return_internal_type=True)
        ct_dev = c_t.get_value(borrow=False, return_internal_type=True)

        f_nnn = function([], [], updates=[(a, (l * a + dot(b, c)))], mode=self.mode)
        f_nnt = function([], [], updates=[(a, (l * a + dot(b, c_t.T)))], mode=self.mode)
        f_ntn = function([], [], updates=[(a, (l * a + dot(b_t.T, c)))], mode=self.mode)
        f_ntt = function(
            [], [], updates=[(a, (l * a + dot(b_t.T, c_t.T)))], mode=self.mode
        )
        f_tnn = function(
            [], [], updates=[(a_t, (l * a_t + dot(b, c).T))], mode=self.mode
        )
        f_tnt = function(
            [], [], updates=[(a_t, (l * a_t + dot(b, c_t.T).T))], mode=self.mode
        )
        f_ttn = function(
            [], [], updates=[(a_t, (l * a_t + dot(b_t.T, c).T))], mode=self.mode
        )
        f_ttt = function(
            [],
            [],
            updates=[(a_t, (l * a_t + dot(b_t.T, c_t.T).T))],
            mode=self.mode,
        )

        # Try with all stride patterns, and all transposed pattern
        for step_signs in product((-1, 1), repeat=6):
            for step in (1, 2):
                a_step1, a_step2, b_step1, b_step2, c_step1, c_step2 = (
                    s * step for s in step_signs
                )

                b.set_value(b_dev.copy()[::b_step1, ::b_step2], borrow=True)
                c.set_value(c_dev.copy()[::c_step1, ::c_step2], borrow=True)
                b_t.set_value(bt_dev.copy()[::b_step2, ::b_step1], borrow=True)
                c_t.set_value(ct_dev.copy()[::c_step2, ::c_step1], borrow=True)

                # Numpy results
                a_n = l * av[::a_step1, ::a_step2] + np.dot(
                    bv[::b_step1, ::b_step2], cv[::c_step1, ::c_step2]
                )
                pt_n = (
                    l * av[::a_step1, ::a_step2].T
                    + np.dot(bv[::b_step1, ::b_step2], cv[::c_step1, ::c_step2]).T
                )

                # a's value is updated, so we need to reinitialize it each time
                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_nnn()
                assert np.allclose(a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_nnt()
                assert np.allclose(a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_ntn()
                assert np.allclose(a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                f_ntt()
                assert np.allclose(a.get_value(), a_n)

                a_t.set_value(
                    np.transpose(a_dev.copy())[::a_step2, ::a_step1], borrow=True
                )
                f_tnn()
                assert np.allclose(a_t.get_value(), pt_n)

                a_t.set_value(
                    np.transpose(a_dev.copy())[::a_step2, ::a_step1], borrow=True
                )
                f_tnt()
                assert np.allclose(a_t.get_value(), pt_n)

                a_t.set_value(
                    np.transpose(a_dev.copy())[::a_step2, ::a_step1], borrow=True
                )
                f_ttn()
                assert np.allclose(a_t.get_value(), pt_n)

                a_t.set_value(
                    np.transpose(a_dev.copy())[::a_step2, ::a_step1], borrow=True
                )
                f_ttt()
                assert np.allclose(a_t.get_value(), pt_n)

    def test_gemm(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        self.cmp_gemm((3, 5), (3, 4), (4, 5), rng)
        self.cmp_gemm((1, 5), (1, 4), (4, 5), rng)
        self.cmp_gemm((3, 1), (3, 4), (4, 1), rng)
        self.cmp_gemm((3, 1), (3, 1), (1, 1), rng)
        self.cmp_gemm((1, 1), (1, 4), (4, 1), rng)
        self.cmp_gemm((3, 5), (3, 1), (1, 5), rng)
        self.cmp_gemm((0, 5), (0, 4), (4, 5), rng)
        self.cmp_gemm((0, 1), (0, 4), (4, 1), rng)
        self.cmp_gemm((0, 5), (0, 1), (1, 5), rng)
        self.cmp_gemm((3, 0), (3, 4), (4, 0), rng)
        self.cmp_gemm((3, 5), (3, 0), (0, 5), rng)
        self.cmp_gemm((0, 0), (0, 4), (4, 0), rng)
        self.cmp_gemm((0, 0), (0, 0), (0, 0), rng)

    def cmp_gemv(self, a_shp, b_shp, c_shp, rng):
        av = self.random(a_shp, rng=rng)
        bv = self.random(*b_shp, rng=rng)
        cv = self.random(c_shp, rng=rng)
        l = np.float32(0.2)

        a = shared(av, "a")
        b = shared(bv, "b")
        c = shared(cv, "c")
        b_t = shared(bv.T, "b.T")

        a_dev = a.get_value(borrow=False, return_internal_type=True)
        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)

        f_n = function([], [], updates=[(a, (a + l * dot(b, c)))], mode=self.mode)

        f_t = function([], [], updates=[(a, (a + l * dot(b_t.T, c)))], mode=self.mode)

        # Try with all stride patterns, and all transposed pattern
        for step_signs in product((1, -1), repeat=4):
            for step in (1, 2):
                a_step, b_step1, b_step2, c_step = (s * step for s in step_signs)

                a.set_value(a_dev.copy()[::a_step], borrow=True)
                b.set_value(b_dev.copy()[::b_step1, ::b_step2], borrow=True)
                # Copy as C so that it becomes F after the transpose in the graph
                b_t.set_value(
                    np.transpose(b_dev).copy(order="C")[::b_step2, ::b_step1],
                    borrow=True,
                )
                c.set_value(c_dev.copy()[::c_step], borrow=True)

                a_n = av[::a_step] + l * np.dot(bv[::b_step1, ::b_step2], cv[::c_step])
                f_n()
                assert np.allclose(a.get_value(), a_n), (a.get_value(), a_n)

                a.set_value(a_dev.copy()[::a_step], borrow=True)
                f_t()
                assert np.allclose(a.get_value(), a_n), (a.get_value(), a_n)

    def test_gemv(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        self.cmp_gemv(3, (3, 5), 5, rng)
        self.cmp_gemv(1, (1, 5), 5, rng)
        self.cmp_gemv(3, (3, 1), 1, rng)
        self.cmp_gemv(1, (1, 1), 1, rng)
        self.cmp_gemv(0, (0, 5), 5, rng)
        self.cmp_gemv(3, (3, 0), 0, rng)
        self.cmp_gemv(0, (0, 1), 1, rng)
        self.cmp_gemv(1, (1, 0), 0, rng)
        self.cmp_gemv(0, (0, 0), 0, rng)

    def cmp_ger(self, a_shp, b_shp, c_shp, rng):
        av = self.random(*a_shp, rng=rng)
        bv = self.random(b_shp, rng=rng)
        cv = self.random(c_shp, rng=rng)
        l = np.float32(0.2)

        a = shared(av, "a")
        b = shared(bv, "b")
        c = shared(cv, "c")
        a_t = shared(av.T, "a.T")

        a_dev = a.get_value(borrow=False, return_internal_type=True)
        b_dev = b.get_value(borrow=False, return_internal_type=True)
        c_dev = c.get_value(borrow=False, return_internal_type=True)

        f_n = function([], [], updates=[(a, (a + l * outer(b, c)))], mode=self.mode)

        f_t = function(
            [], [], updates=[(a_t, (a_t + l * outer(b, c).T))], mode=self.mode
        )

        # Try with all stride patterns, and all transposed patterns
        for step_signs in product((1, -1), repeat=4):
            for step in (1, 2):
                a_step1, a_step2, b_step, c_step = (s * step for s in step_signs)

                a.set_value(a_dev.copy()[::a_step1, ::a_step2], borrow=True)
                a_t.set_value(
                    np.transpose(a_dev.copy())[::a_step1, ::a_step2], borrow=True
                )
                b.set_value(b_dev.copy()[::b_step], borrow=True)
                c.set_value(c_dev.copy()[::c_step], borrow=True)

                f_n()
                n_n = av[::a_step1, ::a_step2] + l * np.outer(
                    bv[::b_step], cv[::c_step]
                )
                assert np.allclose(a.get_value(), n_n), (a.get_value(), n_n)

                f_t()
                n_t = (
                    av.T[::a_step1, ::a_step2]
                    + l * np.outer(bv[::b_step], cv[::c_step]).T
                )
                assert np.allclose(a_t.get_value(), n_t), (a_t.get_value(), n_t)

    def test_ger_strides(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        self.cmp_ger((3, 5), 3, 5, rng)
        self.cmp_ger((1, 5), 1, 5, rng)
        self.cmp_ger((3, 1), 3, 1, rng)
        self.cmp_ger((1, 1), 1, 1, rng)
        self.cmp_ger((0, 5), 0, 5, rng)
        self.cmp_ger((3, 0), 3, 0, rng)
        self.cmp_ger((0, 1), 0, 1, rng)
        self.cmp_ger((1, 0), 1, 0, rng)
        self.cmp_ger((0, 0), 0, 0, rng)

    def test_gemm_non_contiguous(self):
        # test_gemm_non_contiguous: Test if GEMM works well with non-contiguous matrices.
        aval = np.ones((6, 2))
        bval = np.ones((2, 7))
        cval = np.arange(7) + np.arange(0, 0.6, 0.1)[:, np.newaxis]

        a = shared(aval[:3], borrow=True)
        b = shared(bval[:, :5], borrow=True)
        c = shared(cval[:3, :5], borrow=True)

        s = scalar()
        upd_c = s * c + dot(a, b)
        f = function([s], [], updates={c: upd_c})

        f(0)
        ref_output = np.ones((3, 5)) * 2
        unittest_tools.assert_allclose(c.get_value(), ref_output)


class TestInferShape(unittest_tools.InferShapeTester):
    def test_dot22(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        x, y = matrices("xy")
        self._compile_and_check(
            [x, y],
            [_dot22(x, y)],
            [
                rng.random((2, 3)).astype(config.floatX),
                rng.random((3, 4)).astype(config.floatX),
            ],
            Dot22,
        )

    def test_dot22scalar(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        x, y = matrices("xy")
        a = scalar("a")
        self._compile_and_check(
            [x, y, a],
            [_dot22scalar(x, y, a)],
            [
                rng.random((2, 3)).astype(config.floatX),
                rng.random((3, 4)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
            ],
            Dot22Scalar,
        )

    def test_gemm(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        x, y, z = matrices("xyz")
        a = scalar("a")
        b = scalar("b")
        self._compile_and_check(
            [x, y, a, z, b],
            [gemm(z, a, x, y, b)],
            [
                rng.random((2, 3)).astype(config.floatX),
                rng.random((3, 4)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
                rng.random((2, 4)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
            ],
            Gemm,
        )

    def test_gemm_broadcast(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        x, y, z = matrices("xyz")
        a = scalar("a")
        b = scalar("b")

        # Broadcast Z
        self._compile_and_check(
            [x, y, a, z, b],
            [gemm(z, a, x, y, b)],
            [
                rng.random((2, 3)).astype(config.floatX),
                rng.random((3, 4)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
                rng.random((1, 4)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
            ],
            Gemm,
        )

        # Broadcast dot(X, Y)
        self._compile_and_check(
            [x, y, a, z, b],
            [gemm(z, a, x, y, b)],
            [
                rng.random((1, 3)).astype(config.floatX),
                rng.random((3, 4)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
                rng.random((5, 4)).astype(config.floatX),
                np.asarray(1, dtype=config.floatX),
            ],
            Gemm,
        )

    def test_gemv(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        A = matrix("A")
        x, y = vectors("xy")
        a = scalar("a")
        b = scalar("b")
        self._compile_and_check(
            [y, a, A, x, b],
            [gemv(y, a, A, x, b)],
            [
                rng.random((2,)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
                rng.random((2, 3)).astype(config.floatX),
                rng.random((3,)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
            ],
            Gemv,
        )

    def test_ger(self):
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        A = matrix("A")
        x, y = vectors("xy")
        a = scalar("a")
        self._compile_and_check(
            [A, a, x, y],
            [ger(A, a, x, y)],
            [
                rng.random((2, 3)).astype(config.floatX),
                np.asarray(0.5, dtype=config.floatX),
                rng.random((2,)).astype(config.floatX),
                rng.random((3,)).astype(config.floatX),
            ],
            Ger,
        )


rng = np.random.default_rng(unittest_tools.fetch_seed())
TestBatchedDot = makeTester(
    name="BatchedDotTester",
    op=_batched_dot,
    expected=(
        lambda xs, ys: np.asarray(
            [
                x * y if x.ndim == 0 or y.ndim == 0 else np.dot(x, y)
                for x, y in zip(xs, ys, strict=True)
            ],
            dtype=ps.upcast(xs.dtype, ys.dtype),
        )
    ),
    checks={},
    grad=dict(
        correct1=(random(3, 5, 7, rng=rng), random(3, 7, 5, rng=rng)),
        correct2=(random(3, 5, 7, rng=rng), random(3, 7, 9, rng=rng)),
    ),
    good=dict(
        correct1=(random(3, 5, 7, rng=rng), random(3, 7, 5, rng=rng)),
        correct2=(random(3, 5, 7, rng=rng), random(3, 7, 9, rng=rng)),
    ),
    bad_build=dict(
        no_batch_axis2=(random(rng=rng), random(3, 5, rng=rng)),
        no_batch_axis3=(random(3, 5, rng=rng), random(rng=rng)),
    ),
    bad_runtime=dict(
        batch_dim_mismatch1=(random(2, 5, 7, rng=rng), random(3, 7, 9, rng=rng)),
        batch_dim_mismatch2=(random(3, 5, 7, rng=rng), random(2, 7, 9, rng=rng)),
        bad_dim1=(random(3, 5, 7, rng=rng), random(3, 5, 7, rng=rng)),
        bad_dim2=(random(3, 5, 7, rng=rng), random(3, 8, 3, rng=rng)),
    ),
)


def test_batched_dot():
    rng = np.random.default_rng(unittest_tools.fetch_seed())
    first = tensor3("first")
    second = tensor3("second")
    with pytest.warns(FutureWarning):
        output = batched_dot(first, second)
    first_val = rng.random((10, 10, 20)).astype(config.floatX)
    second_val = rng.random((10, 20, 5)).astype(config.floatX)
    result_fn = function([first, second], output)
    result = result_fn(first_val, second_val)
    assert result.shape[0] == first_val.shape[0]
    assert result.shape[1] == first_val.shape[1]
    assert result.shape[2] == second_val.shape[2]

    first_mat = dmatrix("first")
    second_mat = dmatrix("second")
    with pytest.warns(FutureWarning):
        output = batched_dot(first_mat, second_mat)
    first_mat_val = rng.random((10, 10)).astype(config.floatX)
    second_mat_val = rng.random((10, 10)).astype(config.floatX)
    result_fn = function([first_mat, second_mat], output)
    result = result_fn(first_mat_val, second_mat_val)

    assert result.shape[0] == first_mat_val.shape[0]


def test_batched_dot_not_contiguous():
    def np_genarray(*_shape):
        size = 1
        for dimsize in _shape:
            size *= dimsize
        return np.arange(size, dtype=config.floatX).reshape(_shape)

    X = tensor3()
    W = tensor3()
    Z = _batched_dot(X, W)
    f = function([X, W], Z)

    w = np_genarray(30, 10, 5)
    reversed_x_container = np_genarray(20, 40, 30)
    x_container = reversed_x_container.T

    def check_first_dim(inverted):
        direction = -1 if inverted else 1
        x = x_container[::direction, ::2, ::2]
        assert x.shape == (30, 20, 10)
        assert x.strides[0] == direction * np.dtype(config.floatX).itemsize
        assert not (x.flags["C_CONTIGUOUS"] or x.flags["F_CONTIGUOUS"])
        result = f(x, w)
        ref_result = np.asarray([np.dot(u, v) for u, v in zip(x, w, strict=True)])
        utt.assert_allclose(ref_result, result)

    for inverted in (0, 1):
        check_first_dim(inverted)


def test_batched_dot_blas_flags():
    """Test that BatchedDot works regardless of presence of BLAS flags"""
    mode = "FAST_RUN"
    rng = np.random.default_rng(2708)

    x = tensor("x", shape=(2, 5, 3))
    y = tensor("y", shape=(2, 3, 1))
    out = _batched_dot(x, y)
    assert isinstance(out.owner.op, BatchedDot)
    x_test = rng.normal(size=x.type.shape).astype(x.type.dtype)
    y_test = rng.normal(size=y.type.shape).astype(y.type.dtype)

    fn = function([x, y], out, mode=mode)
    [batched_dot_thunk] = fn.vm.thunks
    assert hasattr(batched_dot_thunk, "cthunk")
    np.testing.assert_allclose(fn(x_test, y_test), x_test @ y_test)

    with config.change_flags(blas__ldflags=""):
        fn = function([x, y], out, mode=mode)
    [batched_dot_thunk] = fn.vm.thunks
    assert not hasattr(batched_dot_thunk, "cthunk")
    np.testing.assert_allclose(fn(x_test, y_test), x_test @ y_test)


def test_batched_tensordot():
    rng = np.random.default_rng(unittest_tools.fetch_seed())
    first = tensor4("first")
    second = tensor4("second")
    axes = [[1, 2], [3, 1]]
    with pytest.warns(FutureWarning):
        output = batched_tensordot(first, second, axes)
    first_val = rng.random((8, 10, 20, 3)).astype(config.floatX)
    second_val = rng.random((8, 20, 5, 10)).astype(config.floatX)
    result_fn = function([first, second], output)
    result = result_fn(first_val, second_val)
    assert result.shape[0] == first_val.shape[0]
    assert result.shape[1] == first_val.shape[3]
    assert result.shape[2] == second_val.shape[2]

    first_mat = dmatrix("first")
    second_mat = dmatrix("second")
    axes = 1
    with pytest.warns(FutureWarning):
        output = batched_tensordot(first_mat, second_mat, axes)
    first_mat_val = rng.random((10, 4)).astype(config.floatX)
    second_mat_val = rng.random((10, 4)).astype(config.floatX)
    result_fn = function([first_mat, second_mat], output)
    result = result_fn(first_mat_val, second_mat_val)
    assert result.shape[0] == first_mat_val.shape[0]
    assert len(result.shape) == 1
