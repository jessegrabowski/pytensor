from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile import shared
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.gradient import (
    DisconnectedType,
    Rop,
    disconnected_type,
    grad,
    verify_grad,
)
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.null_type import NullType, null_type
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.utils import MissingInputError
from pytensor.printing import debugprint
from pytensor.tensor.basic import constant
from pytensor.tensor.math import dot, exp, sigmoid
from pytensor.tensor.math import round as pt_round
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.random.utils import RandomStream
from pytensor.tensor.rewriting.shape import ShapeOptimizer
from pytensor.tensor.shape import specify_shape
from pytensor.tensor.type import (
    TensorType,
    dscalars,
    matrices,
    matrix,
    scalar,
    vector,
    vectors,
)
from tests import unittest_tools
from tests.graph.utils import MyVariable


class TestOpFromGraph(unittest_tools.InferShapeTester):
    def test_valid_input(self):
        x, y, z = matrices("xyz")

        with pytest.raises(ValueError, match="Expected at least.*"):
            OpFromGraph([x], [x])()

        with pytest.raises(ValueError, match=r"Expected 1 input\(s\)"):
            OpFromGraph([x], [x]).make_node()

        with pytest.raises(TypeError):
            OpFromGraph((x,), (x,))

        with pytest.raises(TypeError):
            OpFromGraph([1], [1])

        with pytest.raises(NotImplementedError):
            OpFromGraph([x], [x], updates={})

    def test_clone(self):
        x, y, z = matrices("xyz")

        ofg = OpFromGraph([x], [2 * x])

        ofg_clone = ofg.clone()

        assert ofg_clone.fgraph is not ofg.fgraph
        assert ofg_clone.fgraph.outputs != ofg.fgraph.outputs
        assert equal_computations(ofg_clone.fgraph.outputs, ofg.fgraph.outputs)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_straightforward(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        assert np.all(8.0 == fn(xv, yv, zv))
        assert np.all(8.0 == fn(xv, yv, zv))

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_size_changes(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = dot(x, y)
        op = cls_ofg([x, y], [e])
        f = op(x, op(y, z))
        fn = function([x, y, z], f)
        xv = np.ones((2, 3), dtype=config.floatX)
        yv = np.ones((3, 4), dtype=config.floatX) * 3
        zv = np.ones((4, 5), dtype=config.floatX) * 5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert np.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert np.all(180.0 == res)

    def test_overrides_deprecated_api(self):
        inp = scalar("x")
        out = inp + 1
        for kwarg in ("lop_overrides", "grad_overrides", "rop_overrides"):
            with pytest.raises(
                ValueError, match="'default' is no longer a valid value for overrides"
            ):
                OpFromGraph([inp], [out], **{kwarg: "default"})

            with pytest.raises(
                TypeError, match="Variables are no longer valid types for overrides"
            ):
                OpFromGraph([inp], [out], **{kwarg: null_type()})

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_grad(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - grad(pt_sum(f), y)
        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        assert np.all(11.0 == fn(xv, yv, zv))

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_grad_grad(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - grad(pt_sum(f), y)
        f = f - grad(pt_sum(f), y)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        np.testing.assert_array_almost_equal(6.0, fn(xv, yv, zv), 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_shared(self, cls_ofg):
        x, y, z = matrices("xyz")
        s = shared(np.random.random((2, 2)).astype(config.floatX))
        e = x + y * z + s
        op = cls_ofg([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        # print function, function.__module__
        # print fn.maker.fgraph.toposort()
        np.testing.assert_array_almost_equal(8.0, fn(xv, yv, zv), 4)
        np.testing.assert_array_almost_equal(8.0, fn(xv, yv, zv), 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_shared_grad(self, cls_ofg):
        x, y, z = matrices("xyz")
        s = shared(np.random.random((2, 2)).astype(config.floatX))
        e = x + y * z + s
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - grad(pt_sum(f), y)
        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        np.testing.assert_array_almost_equal(11.0 + s.get_value(), fn(xv, yv, zv), 4)

        # grad again the shared variable
        f = op(x, y, z)
        f = f - grad(pt_sum(f), s)
        fn = function([x, y, z], f)
        np.testing.assert_array_almost_equal(15.0 + s.get_value(), fn(xv, yv, zv), 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_grad_override(self, cls_ofg):
        x, y = vectors("xy")

        def go(inps, gs):
            x, y = inps
            (g,) = gs
            return [g * y * 2, g * x * 1.5]

        dedz = vector("dedz")
        op_mul_grad = cls_ofg([x, y, dedz], go([x, y], [dedz]))

        with pytest.warns(FutureWarning, match="grad_overrides is deprecated"):
            op_mul = cls_ofg([x, y], [x * y], grad_overrides=go)
            op_mul2 = cls_ofg([x, y], [x * y], grad_overrides=op_mul_grad)

        # single override case (function or OfG instance)
        xx, yy = vector("xx"), vector("yy")
        for op in [op_mul, op_mul2]:
            zz = pt_sum(op(xx, yy))
            dx, dy = grad(zz, [xx, yy])
            fn = function([xx, yy], [dx, dy])
            xv = np.random.random((16,)).astype(config.floatX)
            yv = np.random.random((16,)).astype(config.floatX)
            dxv, dyv = fn(xv, yv)
            np.testing.assert_array_almost_equal(yv * 2, dxv, 4)
            np.testing.assert_array_almost_equal(xv * 1.5, dyv, 4)

        # list override case
        def go1(inps, gs):
            x, w, b = inps
            g = gs[0]
            return g * w * 2

        def go2(inps, gs):
            x, w, b = inps
            g = gs[0]
            return g * x * 1.5

        w, b = vectors("wb")
        # we make the 3rd gradient default (no override)
        with pytest.warns(FutureWarning, match="grad_overrides is deprecated"):
            op_linear = cls_ofg([x, w, b], [x * w + b], grad_overrides=[go1, go2, None])
        xx, ww, bb = vector("xx"), vector("yy"), vector("bb")
        zz = pt_sum(op_linear(xx, ww, bb))
        dx, dw, db = grad(zz, [xx, ww, bb])
        fn = function([xx, ww, bb], [dx, dw, db])
        xv = np.random.random((16,)).astype(config.floatX)
        wv = np.random.random((16,)).astype(config.floatX)
        bv = np.random.random((16,)).astype(config.floatX)
        dxv, dwv, dbv = fn(xv, wv, bv)
        np.testing.assert_array_almost_equal(wv * 2, dxv, 4)
        np.testing.assert_array_almost_equal(xv * 1.5, dwv, 4)
        np.testing.assert_array_almost_equal(np.ones(16, dtype=config.floatX), dbv, 4)

        # NullType and DisconnectedType
        with pytest.warns(FutureWarning, match="grad_overrides is deprecated"):
            op_linear2 = cls_ofg(
                [x, w, b],
                [x * w + b],
                grad_overrides=[go1, NullType()(), DisconnectedType()()],
                # This is a fake override, so a fake connection_pattern must be provided as well
                connection_pattern=[[True], [True], [False]],
            )
        zz2 = pt_sum(op_linear2(xx, ww, bb))
        dx2, dw2, db2 = grad(
            zz2,
            [xx, ww, bb],
            return_disconnected="Disconnected",
            disconnected_inputs="ignore",
            null_gradients="return",
        )
        assert isinstance(dx2.type, TensorType)
        assert dx2.ndim == 1
        assert isinstance(dw2.type, NullType)
        assert isinstance(db2.type, DisconnectedType)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_lop_override(self, cls_ofg):
        x = vector()
        y = 1.0 / (1.0 + exp(-x))

        def lop_ov(inps, outs, grads):
            (y_,) = outs
            (dedy_,) = grads
            return [2.0 * y_ * (1.0 - y_) * dedy_]

        y_, dedy = vector(), vector()
        op_lop_ov = cls_ofg([x, y_, dedy], [2.0 * y_ * (1.0 - y_) * dedy])

        xx = vector()
        yy1 = pt_sum(sigmoid(xx))
        gyy1 = 2.0 * grad(yy1, xx)

        for ov in [lop_ov, op_lop_ov]:
            op = cls_ofg([x], [y], lop_overrides=ov)
            yy2 = pt_sum(op(xx))
            gyy2 = grad(yy2, xx)
            fn = function([xx], [gyy1, gyy2])

            xval = np.random.random((32,)).astype(config.floatX)
            y1val, y2val = fn(xval)
            np.testing.assert_array_almost_equal(y1val, y2val, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    @pytest.mark.parametrize("use_op_rop_implementation", [True, False])
    def test_rop(self, cls_ofg, use_op_rop_implementation):
        a = vector()
        M = matrix()
        b = dot(a, M)
        op_matmul = cls_ofg([a, M], [b])
        x = vector()
        W = matrix()
        y = op_matmul(x, W)
        du = vector()
        dv = Rop(y, x, du, use_op_rop_implementation=use_op_rop_implementation)
        fn = function([x, W, du], dv)
        xval = np.random.random((16,)).astype(config.floatX)
        Wval = np.random.random((16, 16)).astype(config.floatX)
        duval = np.random.random((16,)).astype(config.floatX)
        dvval = np.dot(duval, Wval)
        dvval2 = fn(xval, Wval, duval)
        np.testing.assert_array_almost_equal(dvval2, dvval, 4)

    @pytest.mark.parametrize("use_op_rop_implementation", [True, False])
    def test_rop_multiple_outputs(self, use_op_rop_implementation):
        a = vector()
        M = matrix()
        b = dot(a, M)
        op_matmul = OpFromGraph([a, M], [b, -b])

        x = vector()
        W = matrix()
        du = vector()

        xval = np.random.random((16,)).astype(config.floatX)
        Wval = np.random.random((16, 16)).astype(config.floatX)
        duval = np.random.random((16,)).astype(config.floatX)

        y = op_matmul(x, W)[0]
        dv = Rop(y, x, du, use_op_rop_implementation=use_op_rop_implementation)
        fn = function([x, W, du], dv)
        result_dvval = fn(xval, Wval, duval)
        expected_dvval = np.dot(duval, Wval)
        np.testing.assert_array_almost_equal(result_dvval, expected_dvval, 4)

        y = op_matmul(x, W)[1]
        dv = Rop(y, x, du, use_op_rop_implementation=use_op_rop_implementation)
        fn = function([x, W, du], dv)
        result_dvval = fn(xval, Wval, duval)
        expected_dvval = -np.dot(duval, Wval)
        np.testing.assert_array_almost_equal(result_dvval, expected_dvval, 4)

        y = pt.add(*op_matmul(x, W))
        dv = Rop(y, x, du, use_op_rop_implementation=use_op_rop_implementation)
        fn = function([x, W, du], dv)
        result_dvval = fn(xval, Wval, duval)
        expected_dvval = np.zeros_like(np.dot(duval, Wval))
        np.testing.assert_array_almost_equal(result_dvval, expected_dvval, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    @pytest.mark.parametrize(
        "use_op_rop_implementation",
        [
            True,
            pytest.param(
                False, marks=pytest.mark.xfail(reason="Custom ROp is ignored")
            ),
        ],
    )
    def test_rop_override(self, cls_ofg, use_op_rop_implementation):
        x, y = vectors("xy")

        def ro(inps, epts):
            x, y = inps
            u, v = epts
            return [u * y * 2.0 + x * v * 1.5]

        u, v = vectors("uv")
        op_mul_rop = cls_ofg([x, y, u, v], ro([x, y], [u, v]))
        op_mul = cls_ofg([x, y], [x * y], rop_overrides=ro)
        op_mul2 = cls_ofg([x, y], [x * y], rop_overrides=op_mul_rop)

        # single override case
        xx, yy = vector("xx"), vector("yy")
        du, dv = vector("du"), vector("dv")
        for op in [op_mul, op_mul2]:
            zz = op_mul(xx, yy)
            dw = Rop(
                zz,
                [xx, yy],
                [du, dv],
                use_op_rop_implementation=use_op_rop_implementation,
            )
            fn = function([xx, yy, du, dv], dw)
            vals = np.random.random((4, 32)).astype(config.floatX)
            dwval = fn(*vals)
            np.testing.assert_array_almost_equal(
                dwval, vals[0] * vals[3] * 1.5 + vals[1] * vals[2] * 2.0, 4
            )

        # TODO list override case

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_connection_pattern_override(self, cls_ofg):
        x, y = vectors("xy")

        def f1(x, y):
            del x
            # but we know how to backpropagate for x for some reasons
            # and we don't care about the gradient wrt y.
            return y + pt_round(y)

        def f1_back(inputs, output_gradients):
            return [output_gradients[0], disconnected_type()]

        with pytest.warns(FutureWarning, match="grad_overrides is deprecated"):
            op = cls_ofg(
                inputs=[x, y],
                outputs=[f1(x, y)],
                grad_overrides=f1_back,
                connection_pattern=[[True], [False]],
                on_unused_input="ignore",
            )

        c = op(x, y)

        g1 = grad(c.sum(), x)

        out = g1.eval(
            {x: np.ones((5,), dtype=np.float32), y: np.ones((5,), dtype=np.float32)}
        )
        np.testing.assert_array_almost_equal(out, [1.0] * 5, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_nested(self, cls_ofg):
        x, y = vectors("xy")
        u, v = x + y, x - y
        op_ft = cls_ofg([x, y], [u, v])
        op_ift = cls_ofg([x, y], [u / 2, v / 2])

        xx, yy = vector("xx"), vector("yy")
        xx2, yy2 = op_ift(*op_ft(xx, yy))
        fn = function([xx, yy], [xx2, yy2])

        xv = np.random.random((16,)).astype(config.floatX)
        yv = np.random.random((16,)).astype(config.floatX)
        xv2, yv2 = fn(xv, yv)
        np.testing.assert_array_almost_equal(xv, xv2, 4)
        np.testing.assert_array_almost_equal(yv, yv2, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_connection_pattern(self, cls_ofg):
        # Basic case
        x, y, z = matrices("xyz")
        out1 = x * y
        out2 = y * z

        op1 = cls_ofg([x, y, z], [out1, out2])
        results = op1.connection_pattern(None)
        expect_result = [[True, False], [True, True], [False, True]]
        assert results == expect_result

        # Graph with ops that don't have a 'full' connection pattern
        # and with ops that have multiple outputs
        m, n, p, q = matrices("mnpq")
        o1, o2 = op1(m, n, p)
        out1, out2 = op1(o1, q, o2)
        op2 = cls_ofg([m, n, p, q], [out1, out2])

        results = op2.connection_pattern(None)
        expect_result = [[True, False], [True, True], [False, True], [True, True]]
        assert results == expect_result

        # Inner graph where some computation doesn't rely on explicit inputs
        srng = RandomStream(seed=234)
        rv_u = srng.uniform((2, 2))
        x, y = matrices("xy")
        out1 = x + rv_u
        out2 = y + 3
        out3 = 3 + rv_u
        op3 = cls_ofg([x, y], [out1, out2, out3])

        results = op3.connection_pattern(None)
        expect_result = [
            [True, False, False],
            [False, True, False],
            [True, False, True],
        ]
        assert results == expect_result

    def test_infer_shape(self):
        # test infer shape does not need to against inline case
        # since the Op is remove during optimization phase
        x = matrix("x")
        y = matrix("y")
        o1 = x + y
        o2 = x * y
        op_graph = OpFromGraph([x, y], [o1, o2])

        q = matrix("q")
        p = matrix("p")
        self._compile_and_check(
            [q, p],
            op_graph(q, p),
            [
                np.ones([3, 4], dtype=config.floatX),
                np.ones([3, 4], dtype=config.floatX),
            ],
            OpFromGraph,
        )

        # Make sure `OpFromGraph.infer_shape` can handle objects without a
        # shape
        x = MyVariable("x")
        y = matrix("y")
        z = specify_shape(vector("z"), (2,))

        op_graph = OpFromGraph([x, y, z], [x, y])

        op_var = op_graph(x, y, z)

        fg = FunctionGraph(outputs=[op_var[1]], clone=False)
        opt_res = rewrite_graph(fg, custom_rewrite=ShapeOptimizer())

        assert opt_res.shape_feature.shape_of[x] is None
        assert opt_res.shape_feature.shape_of[z][0].data == 2

    @config.change_flags(compute_test_value="raise")
    def test_compute_test_value(self):
        x = scalar("x")
        x.tag.test_value = np.array(1.0, dtype=config.floatX)
        op = OpFromGraph([x], [x**3])
        y = scalar("y")
        y.tag.test_value = np.array(1.0, dtype=config.floatX)
        f = op(y)
        grad_f = grad(f, y)
        assert grad_f.tag.test_value is not None

    def test_make_node_shared(self):
        """Make sure we can provide `OpFromGraph.make_node` new shared inputs and get a valid `OpFromGraph`."""

        x = pt.scalar("x")
        y = shared(1.0, name="y")

        test_ofg = OpFromGraph([x], [x + y], on_unused_input="ignore")
        assert test_ofg.shared_inputs == [y]

        out = test_ofg(x)

        y_clone = y.clone()
        assert y_clone != y
        y_clone.name = "y_clone"

        out_new = test_ofg.make_node(*(out.owner.inputs[:1] + [y_clone])).outputs[0]

        assert "on_unused_input" in out_new.owner.op.kwargs
        assert out_new.owner.op.shared_inputs == [y_clone]

        out_fn = function([x], out_new)
        assert np.array_equal(out_fn(1.0), 2.0)

        y_clone.set_value(2.0)
        assert np.array_equal(out_fn(1.0), 3.0)

        # This should also work, because the containers are the same:
        # y.set_value(1.0)
        # assert np.array_equal(out_fn(1.0), 2.0)

    def test_shared_with_constant_input(self):
        """Make sure that a constant input can be given to an `OpFromGraph` instance."""
        x = pt.scalar("x")
        y = shared(1.0, name="y")

        test_ofg = OpFromGraph([x], [x + y])
        assert test_ofg.shared_inputs == [y]

        out = test_ofg(pt.as_tensor(1.0, dtype=config.floatX))

        out_fn = function([], out)
        assert np.array_equal(out_fn(), 2.0)

    def test_missing_input(self):
        x = pt.lscalar("x")

        with pytest.raises(MissingInputError):
            OpFromGraph([], [x])

    def test_shared_to_nonshared_input(self):
        """Make sure that shared variables can be replaced with non-shared variables."""
        x = pt.scalar("x")
        y = shared(1.0, name="y")

        test_ofg = OpFromGraph([], [y])
        assert test_ofg.shared_inputs == [y]

        out_1_fn = function([], test_ofg())
        res_1 = out_1_fn()

        assert np.array_equal(res_1, 1.0)

        test_ofg_new = test_ofg.make_node(x)
        assert test_ofg_new.op.shared_inputs == []

        out_2_fn = function([x], test_ofg_new.outputs[0])
        res_2 = out_2_fn(np.array(1.0, dtype=config.floatX))

        assert np.array_equal(res_2, 1.0)

    def test_outputs_consistency(self):
        """Make sure that `OpFromGraph.fn` doesn't change the value of `OpFromGraph.inner_outputs`."""

        x = scalar("x")
        op = OpFromGraph([x], [x**2 / x], mode="FAST_RUN")

        # Confirm that the inner-graph is as expected
        assert equal_computations(op.inner_outputs, [x**2 / x], op.inner_inputs, [x])

        # These outputs of the compiled `op.fgraph` should differ from the
        # original, uncompiled `op.fgraph` outputs
        fn = op.fn
        new_inputs = fn.maker.fgraph.inputs
        new_outputs = fn.maker.fgraph.outputs
        assert not equal_computations(new_outputs, [x**2 / x], new_inputs, [x])

        # The original `op.fgraph` outputs should stay the same, though
        assert equal_computations(op.inner_outputs, [x**2 / x], op.inner_inputs, [x])

    def test_explicit_input_from_constant(self):
        x = pt.dscalar("x")
        y = constant(1.0, dtype=x.type.dtype, name="y")
        test_ofg = OpFromGraph([x, y], [x + y])

        out = test_ofg(x, y)
        assert out.eval({x: 5}) == 6

        out = test_ofg(x, x)
        assert out.eval({x: 5}) == 10

    def test_explicit_input_from_shared(self):
        x = pt.dscalar("x")
        y = shared(1.0, name="y")

        with pytest.raises(
            ValueError,
            match=r"The inner-graph implicitly depends on the following shared variables \[y\]",
        ):
            OpFromGraph([x], [x + y], strict=True)

        test_ofg = OpFromGraph([x, y], [x + y], strict=True)

        out = test_ofg(x, y)
        assert out.eval({x: 5}) == 6
        y.set_value(2.0)
        assert out.eval({x: 6}) == 8

        out = test_ofg(y, y)
        assert out.eval() == 4

    def test_L_op_disconnected_output_grad(self):
        x, y = dscalars("x", "y")
        rng = np.random.default_rng(594)
        point = list(rng.normal(size=(2,)))

        out1 = x + y
        out2 = x * y
        out3 = out1 * out2  # Create dependency between outputs
        op = OpFromGraph([x, y], [out1, out2, out3])
        verify_grad(lambda x, y: pt.add(*op(x, y)), point, rng=rng)
        verify_grad(lambda x, y: pt.add(*op(x, y)[:-1]), point, rng=rng)
        verify_grad(lambda x, y: pt.add(*op(x, y)[1:]), point, rng=rng)
        verify_grad(lambda x, y: pt.add(*op(x, y)[::2]), point, rng=rng)
        verify_grad(lambda x, y: op(x, y)[0], point, rng=rng)
        verify_grad(lambda x, y: op(x, y)[1], point, rng=rng)
        verify_grad(lambda x, y: op(x, y)[2], point, rng=rng)

        # Test disconnected graphs are handled correctly
        op = OpFromGraph([x, y], [x**2, y**3])
        with pytest.warns(UserWarning):
            grad_x_wrt_y = grad(
                op(x, y)[0],
                wrt=y,
                return_disconnected="disconnected",
                disconnected_inputs="warn",
            )
            assert isinstance(grad_x_wrt_y.type, DisconnectedType)

    def test_repeated_inputs(self):
        x = pt.dscalar("x")
        y = pt.dscalar("y")

        with pytest.raises(
            ValueError,
            match="The following variables were provided more than once as inputs to the "
            "OpFromGraph",
        ):
            OpFromGraph([x, x, y], [x + y])

        # Test that repeated inputs will be allowed if unused inputs are ignored
        g = OpFromGraph([x, x, y], [x + y], on_unused_input="ignore")
        f = g(x, x, y)
        assert f.eval({x: 5, y: 5}) == 10


@config.change_flags(floatX="float64")
def test_debugprint():
    x, y, z = matrices("xyz")
    e = x + y * z
    op = OpFromGraph([x, y, z], [e])
    out = op(x, y, z)

    output_str = debugprint(out, file="str")
    lines = output_str.split("\n")

    exp_res = """OpFromGraph{inline=False} [id A]
 ├─ x [id B]
 ├─ y [id C]
 └─ z [id D]

Inner graphs:

OpFromGraph{inline=False} [id A]
 ← Add [id E]
    ├─ *0-<Matrix(float64, shape=(?, ?))> [id F]
    └─ Mul [id G]
       ├─ *1-<Matrix(float64, shape=(?, ?))> [id H]
       └─ *2-<Matrix(float64, shape=(?, ?))> [id I]
"""

    for truth, out in zip(exp_res.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()
