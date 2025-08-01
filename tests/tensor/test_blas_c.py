from warnings import warn

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.blas import Ger
from pytensor.tensor.blas_c import CGemv, CGer, must_initialize_y_gemv
from pytensor.tensor.type import dmatrix, dvector, matrix, scalar, tensor, vector
from tests import unittest_tools
from tests.tensor.test_blas import BaseGemv, TestBlasStrides
from tests.unittest_tools import OptimizationTestMixin


mode_blas_opt = pytensor.compile.get_default_mode().including(
    "BlasOpt", "specialize", "InplaceBlasOpt", "c_blas"
)


def skip_if_blas_ldflags_empty(*functions_detected):
    if pytensor.config.blas__ldflags == "":
        functions_string = ""
        if functions_detected:
            functions_string = " (at least " + (", ".join(functions_detected)) + ")"
        pytest.skip(
            "This test is useful only when PyTensor can access to BLAS functions"
            + functions_string
            + " other than [sd]gemm_."
        )


class TestCGer(OptimizationTestMixin):
    def setup_method(self):
        self.manual_setup_method()

    def manual_setup_method(self, dtype="float64"):
        # This tests can run even when pytensor.config.blas__ldflags is empty.
        self.dtype = dtype
        self.mode = pytensor.compile.get_default_mode().including("fast_run")
        self.A = tensor(dtype=dtype, shape=(None, None))
        self.a = tensor(dtype=dtype, shape=())
        self.x = tensor(dtype=dtype, shape=(None,))
        self.y = tensor(dtype=dtype, shape=(None,))
        self.Aval = np.ones((2, 3), dtype=dtype)
        self.xval = np.asarray([1, 2], dtype=dtype)
        self.yval = np.asarray([1.5, 2.7, 3.9], dtype=dtype)

    def function(self, inputs, outputs):
        return pytensor.function(
            inputs,
            outputs,
            mode=self.mode,
            # allow_inplace=True,
        )

    def run_f(self, f):
        f(self.Aval, self.xval, self.yval)
        f(self.Aval[::-1, ::-1], self.xval, self.yval)

    def b(self, bval):
        return pt.as_tensor_variable(np.asarray(bval, dtype=self.dtype))

    def test_eq(self):
        assert CGer(True) == CGer(True)
        assert CGer(False) == CGer(False)
        assert CGer(False) != CGer(True)

        assert CGer(True) != Ger(True)
        assert CGer(False) != Ger(False)

        # assert that eq works for non-CGer instances
        assert CGer(False) is not None
        assert CGer(True) is not None

    def test_hash(self):
        assert hash(CGer(True)) == hash(CGer(True))
        assert hash(CGer(False)) == hash(CGer(False))
        assert hash(CGer(False)) != hash(CGer(True))

    def test_optimization_pipeline(self):
        skip_if_blas_ldflags_empty()
        f = self.function([self.x, self.y], pt.outer(self.x, self.y))
        self.assertFunctionContains(f, CGer(destructive=True))
        f(self.xval, self.yval)  # DebugMode tests correctness

    def test_optimization_pipeline_float(self):
        skip_if_blas_ldflags_empty()
        self.manual_setup_method("float32")
        f = self.function([self.x, self.y], pt.outer(self.x, self.y))
        self.assertFunctionContains(f, CGer(destructive=True))
        f(self.xval, self.yval)  # DebugMode tests correctness

    def test_int_fails(self):
        self.manual_setup_method("int32")
        f = self.function([self.x, self.y], pt.outer(self.x, self.y))
        self.assertFunctionContains0(f, CGer(destructive=True))
        self.assertFunctionContains0(f, CGer(destructive=False))

    def test_A_plus_outer(self):
        skip_if_blas_ldflags_empty()
        f = self.function([self.A, self.x, self.y], self.A + pt.outer(self.x, self.y))
        self.assertFunctionContains(f, CGer(destructive=False))
        self.run_f(f)  # DebugMode tests correctness

    def test_A_plus_scaled_outer(self):
        skip_if_blas_ldflags_empty()
        f = self.function(
            [self.A, self.x, self.y], self.A + 0.1 * pt.outer(self.x, self.y)
        )
        self.assertFunctionContains(f, CGer(destructive=False))
        self.run_f(f)  # DebugMode tests correctness


class TestCGemv(OptimizationTestMixin):
    """
    Tests of CGemv specifically.

    Generic tests of Gemv-compatibility, including both dtypes are
    done below in TestCGemvFloat32 and TestCGemvFloat64
    """

    def setup_method(self):
        # This tests can run even when pytensor.config.blas__ldflags is empty.
        dtype = "float64"
        self.dtype = dtype
        self.mode = pytensor.compile.get_default_mode().including("fast_run")
        # matrix
        self.A = tensor("A", dtype=dtype, shape=(None, None))
        self.Aval = np.ones((2, 3), dtype=dtype)

        # vector
        self.x = tensor("x", dtype=dtype, shape=(None,))
        self.y = tensor("y", dtype=dtype, shape=(None,))
        self.xval = np.asarray([1, 2], dtype=dtype)
        self.yval = np.asarray([1.5, 2.7, 3.9], dtype=dtype)

        # scalar
        self.a = tensor("a", dtype=dtype, shape=())

    @pytest.mark.parametrize("inplace", [True, False])
    def test_nan_beta_0(self, inplace):
        mode = self.mode.including()
        mode.check_isfinite = False
        f = pytensor.function(
            [self.A, self.x, pytensor.In(self.y, mutable=inplace), self.a],
            self.a * self.y + pt.dot(self.A, self.x),
            mode=mode,
        )
        [node] = f.maker.fgraph.apply_nodes
        assert isinstance(node.op, CGemv) and node.op.inplace == inplace
        for rows in (3, 1):
            Aval = np.ones((rows, 1), dtype=self.dtype)
            xval = np.ones((1,), dtype=self.dtype)
            yval = np.full((rows,), np.nan, dtype=self.dtype)
            zval = f(Aval, xval, yval, 0)
            assert not np.isnan(zval).any()

    def test_optimizations_vm(self):
        skip_if_blas_ldflags_empty()
        """ Test vector dot matrix """
        f = pytensor.function([self.x, self.A], pt.dot(self.x, self.A), mode=self.mode)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, pt.dot)
        self.assertFunctionContains1(f, CGemv(inplace=True))

        # Assert they produce the same output
        assert np.allclose(f(self.xval, self.Aval), np.dot(self.xval, self.Aval))

        # Test with negative strides on 2 dims
        assert np.allclose(
            f(self.xval, self.Aval[::-1, ::-1]),
            np.dot(self.xval, self.Aval[::-1, ::-1]),
        )

    def test_optimizations_mv(self):
        skip_if_blas_ldflags_empty()
        """ Test matrix dot vector """
        f = pytensor.function([self.A, self.y], pt.dot(self.A, self.y), mode=self.mode)

        # Assert that the dot was optimized somehow
        self.assertFunctionContains0(f, pt.dot)
        self.assertFunctionContains1(f, CGemv(inplace=True))

        # Assert they produce the same output
        assert np.allclose(f(self.Aval, self.yval), np.dot(self.Aval, self.yval))
        # Test with negative strides on 2 dims
        assert np.allclose(
            f(self.Aval[::-1, ::-1], self.yval),
            np.dot(self.Aval[::-1, ::-1], self.yval),
        )

    def test_must_initialize_y_gemv(self):
        if must_initialize_y_gemv():
            # FIME: This warn should be emitted by the function if we find it relevant
            # Not in a test that doesn't care about the outcome either way
            warn(
                "WARNING: The current BLAS requires PyTensor to initialize"
                " memory for some GEMV calls which will result in a minor"
                " degradation in performance for such calls."
            )

    def t_gemv1(self, m_shp):
        """test vector2 + dot(matrix, vector1)"""
        rng = np.random.default_rng(unittest_tools.fetch_seed())
        v1 = pytensor.shared(np.array(rng.uniform(size=(m_shp[1],)), dtype="float32"))
        v2_orig = np.array(rng.uniform(size=(m_shp[0],)), dtype="float32")
        v2 = pytensor.shared(v2_orig)
        m = pytensor.shared(np.array(rng.uniform(size=m_shp), dtype="float32"))

        f = pytensor.function([], v2 + pt.dot(m, v1), mode=self.mode)

        # Assert they produce the same output
        assert np.allclose(f(), np.dot(m.get_value(), v1.get_value()) + v2_orig)
        topo = [n.op for n in f.maker.fgraph.toposort()]
        assert topo == [CGemv(inplace=False)], topo

        # test the inplace version
        g = pytensor.function(
            [], [], updates=[(v2, v2 + pt.dot(m, v1))], mode=self.mode
        )

        # Assert they produce the same output
        g()
        assert np.allclose(
            v2.get_value(), np.dot(m.get_value(), v1.get_value()) + v2_orig
        )
        topo = [n.op for n in g.maker.fgraph.toposort()]
        assert topo == [CGemv(inplace=True)]

        # Do the same tests with a matrix with strides in both dimensions
        m.set_value(m.get_value(borrow=True)[::-1, ::-1], borrow=True)
        v2.set_value(v2_orig)
        assert np.allclose(f(), np.dot(m.get_value(), v1.get_value()) + v2_orig)
        g()
        assert np.allclose(
            v2.get_value(), np.dot(m.get_value(), v1.get_value()) + v2_orig
        )

    def test_gemv1(self):
        skip_if_blas_ldflags_empty()
        self.t_gemv1((3, 2))
        self.t_gemv1((1, 2))
        self.t_gemv1((0, 2))
        self.t_gemv1((3, 1))
        self.t_gemv1((3, 0))
        self.t_gemv1((1, 1))
        self.t_gemv1((1, 0))
        self.t_gemv1((0, 1))
        self.t_gemv1((0, 0))

    def test_gemv_dimensions(self, dtype="float32"):
        alpha = pytensor.shared(np.asarray(1.0, dtype=dtype), name="alpha")
        beta = pytensor.shared(np.asarray(1.0, dtype=dtype), name="beta")

        z = beta * self.y + alpha * pt.dot(self.A, self.x)
        f = pytensor.function([self.A, self.x, self.y], z, mode=self.mode)

        # Matrix value
        A_val = np.ones((5, 3), dtype=dtype)
        # Different vector length
        ones_3 = np.ones(3, dtype=dtype)
        ones_4 = np.ones(4, dtype=dtype)
        ones_5 = np.ones(5, dtype=dtype)
        ones_6 = np.ones(6, dtype=dtype)

        f(A_val, ones_3, ones_5)
        f(A_val[::-1, ::-1], ones_3, ones_5)
        with pytest.raises(ValueError):
            f(A_val, ones_4, ones_5)
        with pytest.raises(ValueError):
            f(A_val, ones_3, ones_6)
        with pytest.raises(ValueError):
            f(A_val, ones_4, ones_6)

    def test_multiple_inplace(self):
        skip_if_blas_ldflags_empty()
        x = dmatrix("x")
        y = dvector("y")
        z = dvector("z")
        f = pytensor.function(
            [x, y, z], [pt.dot(y, x), pt.dot(z, x)], mode=mode_blas_opt
        )
        vx = np.random.random((3, 3))
        vy = np.random.random(3)
        vz = np.random.random(3)
        out = f(vx, vy, vz)
        assert np.allclose(out[0], np.dot(vy, vx))
        assert np.allclose(out[1], np.dot(vz, vx))
        assert (
            len([n for n in f.maker.fgraph.apply_nodes if isinstance(n.op, AllocEmpty)])
            == 2
        )


class TestCGemvFloat32(BaseGemv, OptimizationTestMixin):
    mode = mode_blas_opt
    dtype = "float32"
    gemv = CGemv(inplace=False)
    gemv_inplace = CGemv(inplace=True)

    def setup_method(self):
        skip_if_blas_ldflags_empty()


class TestCGemvFloat64(BaseGemv, OptimizationTestMixin):
    mode = mode_blas_opt
    dtype = "float64"
    gemv = CGemv(inplace=False)
    gemv_inplace = CGemv(inplace=True)

    def setup_method(self):
        skip_if_blas_ldflags_empty()


class TestCGemvNoFlags:
    mode = mode_blas_opt
    gemv = CGemv(inplace=False)
    M = 4
    N = 5
    slice_step = 3

    def get_function(self, dtype, transpose_A=False, slice_tensors=False):
        alpha = scalar(dtype=dtype)
        beta = scalar(dtype=dtype)
        A = matrix(dtype=dtype)
        x = vector(dtype=dtype)
        y = vector(dtype=dtype)
        if transpose_A:
            A_1 = A.T
        else:
            A_1 = A
        if slice_tensors:
            A_2 = A_1[:: -self.slice_step]
            x_2 = x[:: -self.slice_step]
            y_2 = y[:: -self.slice_step]
        else:
            A_2 = A_1
            x_2 = x
            y_2 = y
        return pytensor.function(
            [alpha, A, x, beta, y],
            self.gemv(y_2, alpha, A_2, x_2, beta),
            mode=self.mode,
        )

    def get_data(self, dtype, alpha, beta, transpose_A=False, slice_tensors=False):
        if slice_tensors:
            if transpose_A:
                A_shape = (self.N, self.M * self.slice_step)
            else:
                A_shape = (self.M * self.slice_step, self.N)
            x_shape = (self.N * self.slice_step,)
            y_shape = (self.M * self.slice_step,)
        else:
            if transpose_A:
                A_shape = (self.N, self.M)
            else:
                A_shape = (self.M, self.N)
            x_shape = (self.N,)
            y_shape = (self.M,)
        A = np.random.random(A_shape).astype(dtype)
        x = np.random.random(x_shape).astype(dtype)
        y = np.random.random(y_shape).astype(dtype)
        return (alpha, A, x, beta, y)

    def compute_ref(self, alpha, A, x, beta, y, transpose_A, slice_tensors):
        if transpose_A:
            A = A.T
        if slice_tensors:
            A = A[:: -self.slice_step]
            x = x[:: -self.slice_step]
            y = y[:: -self.slice_step]
        ref_val = alpha * np.dot(A, x)
        if beta != 0:
            ref_val += beta * y
        return ref_val

    @pytensor.config.change_flags(blas__ldflags="")
    def run_cgemv(self, dtype, ALPHA, BETA, transpose_A, slice_tensors):
        f = self.get_function(
            dtype, transpose_A=transpose_A, slice_tensors=slice_tensors
        )
        values = self.get_data(
            dtype, ALPHA, BETA, transpose_A=transpose_A, slice_tensors=slice_tensors
        )
        assert any(isinstance(node.op, CGemv) for node in f.maker.fgraph.apply_nodes)
        z_val = f(*values)
        assert z_val.dtype == dtype
        assert z_val.ndim == 1
        assert z_val.shape[0] == self.M
        ref_val = self.compute_ref(*((*values, transpose_A, slice_tensors)))
        unittest_tools.assert_allclose(ref_val, z_val)

    def test_cgemv(self):
        for dtype in ("float32", "float64"):
            for alpha in (0, 1, -2):
                for beta in (0, 1, -2):
                    for transpose_A in (False, True):
                        for slice_tensors in (False, True):
                            self.run_cgemv(
                                dtype,
                                alpha,
                                beta,
                                transpose_A,
                                slice_tensors,
                            )


class TestSdotNoFlags(TestCGemvNoFlags):
    M = 1


class TestBlasStridesC(TestBlasStrides):
    mode = mode_blas_opt


def test_gemv_vector_dot_perf(benchmark):
    n = 400_000
    a = pt.vector("A", shape=(n,))
    b = pt.vector("x", shape=(n,))

    out = CGemv(inplace=True)(
        pt.empty((1,)),
        1.0,
        a[None],
        b,
        0.0,
    )
    fn = pytensor.function([a, b], out, accept_inplace=True, trust_input=True)

    rng = np.random.default_rng(430)
    test_a = rng.normal(size=n)
    test_b = rng.normal(size=n)

    np.testing.assert_allclose(
        fn(test_a, test_b),
        np.dot(test_a, test_b),
    )

    benchmark(fn, test_a, test_b)


@pytest.mark.parametrize(
    "neg_stride1", (True, False), ids=["neg_stride1", "pos_stride1"]
)
@pytest.mark.parametrize(
    "neg_stride0", (True, False), ids=["neg_stride0", "pos_stride0"]
)
@pytest.mark.parametrize("F_layout", (True, False), ids=["F_layout", "C_layout"])
def test_gemv_negative_strides_perf(neg_stride0, neg_stride1, F_layout, benchmark):
    A = pt.matrix("A", shape=(512, 512))
    x = pt.vector("x", shape=(A.type.shape[-1],))
    y = pt.vector("y", shape=(A.type.shape[0],))

    out = CGemv(inplace=False)(
        y,
        1.0,
        A,
        x,
        1.0,
    )
    fn = pytensor.function([A, x, y], out, trust_input=True)

    rng = np.random.default_rng(430)
    test_A = rng.normal(size=A.type.shape)
    test_x = rng.normal(size=x.type.shape)
    test_y = rng.normal(size=y.type.shape)

    if F_layout:
        test_A = test_A.T
    if neg_stride0:
        test_A = test_A[::-1]
    if neg_stride1:
        test_A = test_A[:, ::-1]
    assert (test_A.strides[0] < 0) == neg_stride0
    assert (test_A.strides[1] < 0) == neg_stride1

    # Check result is correct by using a copy of A with positive strides
    res = fn(test_A, test_x, test_y)
    np.testing.assert_allclose(res, fn(test_A.copy(), test_x, test_y))

    benchmark(fn, test_A, test_x, test_y)


@pytest.mark.parametrize("inplace", (True, False), ids=["inplace", "no_inplace"])
@pytest.mark.parametrize("n", [2**7, 2**9, 2**13])
def test_ger_benchmark(n, inplace, benchmark):
    alpha = pt.dscalar("alpha")
    x = pt.dvector("x")
    y = pt.dvector("y")
    A = pt.dmatrix("A")

    out = alpha * pt.outer(x, y) + A

    fn = pytensor.function(
        [alpha, x, y, pytensor.In(A, mutable=inplace)], out, trust_input=True
    )

    rng = np.random.default_rng([2274, n])
    alpha_test = rng.normal(size=())
    x_test = rng.normal(size=(n,))
    y_test = rng.normal(size=(n,))
    A_test = rng.normal(size=(n, n))

    benchmark(fn, alpha_test, x_test, y_test, A_test)
