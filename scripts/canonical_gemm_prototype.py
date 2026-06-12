"""Prototype: CanonicalGEMM — a clean, backend-neutral fused matmul-accumulate op.

    out = alpha * (A @ B) + beta * C

Design (vs the stock C-ABI `Gemm`):
  - C may broadcast to the matmul output (vector bias, row, matrix, scalar). No rank-2
    accumulator constraint -> the bias-fusion "broadcasting problem" disappears by design.
  - No `inplace` flag in the graph op; in-place is a codegen decision, not an IR variant.
  - Differentiable: its gradients are themselves CanonicalGEMMs (closed under autodiff),
    mirroring how Gemm's grads are Dot22/Gemm.
  - One matmul-anchored rewrite introduces it; each backend lowers it as it likes
    (here: naive dot+add on perform/numba/jax; a C backend would lower to BLAS ?gemm).

Run: python scratch_canonical_gemm.py
"""
# ruff: noqa: E402, T201  prototype script: prints are intentional; jax x64 set before submodule import

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.scalar import upcast
from scipy.linalg.blas import get_blas_funcs
from pytensor.tensor.blas import Dot22
from pytensor.tensor.math import Dot, _matmul
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.type import TensorType

pytensor.config.floatX = "float64"


def _reduce_to(x, var):
    """Sum x (shape of the matmul output) down to var's shape, undoing broadcasting."""
    extra = x.type.ndim - var.type.ndim
    if extra:
        x = x.sum(axis=tuple(range(extra)))
    keep = tuple(i for i, b in enumerate(var.type.broadcastable) if b)
    if keep:
        x = x.sum(axis=keep, keepdims=True)
    return x


class CanonicalGEMM(Op):
    # Mirrors cBLAS ?gemm: A, B are const operands; C is the only read/write buffer.
    # overwrite_c is the sole destructive flag (A/B can't be overwritten — they're const).
    __props__ = ("overwrite_c",)

    def __init__(self, overwrite_c=False):
        self.overwrite_c = overwrite_c
        if overwrite_c:
            self.destroy_map = {0: [2]}  # output reuses C's buffer (index 2)

    def make_node(self, A, B, C, alpha, beta):
        A, B, C, alpha, beta = (
            pt.as_tensor_variable(v) for v in (A, B, C, alpha, beta)
        )
        if A.type.ndim != 2 or B.type.ndim != 2:
            raise TypeError("CanonicalGEMM: A and B must be 2-D")
        if alpha.type.ndim != 0 or beta.type.ndim != 0:
            raise TypeError("CanonicalGEMM: alpha and beta must be scalars")
        dtype = upcast(A.dtype, B.dtype, C.dtype, alpha.dtype, beta.dtype)
        # propagate static shapes: out is (A.rows, B.cols), carrying known dims through
        out = TensorType(dtype, shape=(A.type.shape[0], B.type.shape[1]))()
        return Apply(self, [A, B, C, alpha, beta], [out])

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        """Enable overwrite_c iff the accumulator C (input 2) is lifetime-safe to destroy.

        The inplace pass only offers inputs with no other consumers. C is the only valid
        in-place target (A, B are const; only C shares the (m, n) output shape).
        """
        if self.overwrite_c or 2 not in allowed_inplace_inputs:
            return self
        return type(self)(overwrite_c=True)

    def infer_shape(self, fgraph, node, ishapes):
        return [(ishapes[0][0], ishapes[1][1])]

    def perform(self, node, inputs, out):
        A, B, C, alpha, beta = inputs
        gemm = get_blas_funcs(
            "gemm", (A, B)
        )  # right xgemm for the dtype (s/d/c/z gemm)
        if self.overwrite_c:
            # fused in-place BLAS gemm: C := alpha*A@B + beta*C, reusing C's buffer (when F-contiguous)
            out[0][0] = gemm(alpha, A, B, beta=beta, c=C, overwrite_c=1)
        elif C.ndim == 2 and C.shape == (A.shape[0], B.shape[1]):
            out[0][0] = gemm(
                alpha, A, B, beta=beta, c=C
            )  # full-shape accumulator, no overwrite
        else:
            # C broadcasts (vector/row bias): gemm can't take it as `c`, so add it after the matmul
            out[0][0] = np.asarray(
                gemm(alpha, A, B) + beta * C, dtype=node.outputs[0].type.dtype
            )

    def pullback(
        self, inputs, outputs, output_cotangents
    ):  # modern grad hook (replaces L_op)
        A, B, C, alpha, beta = inputs
        (g,) = output_cotangents
        zero = pt.zeros((), dtype=g.type.dtype)
        dA = canonical_gemm(g, B.T, zero, alpha, zero)  # alpha * g @ B.T
        dB = canonical_gemm(A.T, g, zero, alpha, zero)  # alpha * A.T @ g
        AB = pt.dot(A, B)
        dalpha = pt.sum(g * AB)
        dbeta = pt.sum(g * pt.broadcast_to(C, (A.shape[0], B.shape[1])))
        dC = _reduce_to(beta * g, C)
        return [dA, dB, dC, dalpha, dbeta]


canonical_gemm = CanonicalGEMM()


# --- backend lowerings (naive for now; a C backend would emit a BLAS ?gemm call) ---
@numba_funcify.register(CanonicalGEMM)
def _numba(op, node=None, **kwargs):
    @numba_njit
    def cgemm(A, B, C, alpha, beta):
        return (
            alpha * np.dot(np.ascontiguousarray(A), np.ascontiguousarray(B)) + beta * C
        )

    return cgemm


@jax_funcify.register(CanonicalGEMM)
def _jax(op, **kwargs):
    def cgemm(A, B, C, alpha, beta):
        return alpha * jnp.dot(A, B) + beta * C

    return cgemm


def _is_scalar_like(v):
    # rank-0, or a broadcasted scalar (e.g. ExpandDims{(0,1)}(2.5) -> (1,1), all-broadcastable)
    return v.type.ndim == 0 or all(v.type.broadcastable)


def _peel_scalar(v):
    """If v == s * w with s scalar-like (and w the lone non-scalar factor), return (s_rank0, w); else (None, v)."""
    o = v.owner
    if o is not None and o.op == pt.mul:
        scalars = [i for i in o.inputs if _is_scalar_like(i)]
        nonscalars = [i for i in o.inputs if not _is_scalar_like(i)]
        if scalars and len(nonscalars) == 1:
            s = scalars[0] if len(scalars) == 1 else pt.mul(*scalars)
            if s.type.ndim != 0:
                s = (
                    s.squeeze()
                )  # drop the broadcast dims to recover a rank-0 alpha/beta
            return s, nonscalars[0]
    return None, v


@register_specialize
@node_rewriter([Dot, Dot22, _matmul])  # anchor on the rare matmul, not on every add
def fuse_canonical_gemm(fgraph, node):
    """alpha*(A@B) + beta*C  ->  CanonicalGEMM(A, B, C, alpha, beta), looking up the scalars."""
    A, B = node.inputs[0], node.inputs[1]
    if A.type.ndim != 2 or B.type.ndim != 2:
        return None
    mm = node.outputs[0]
    if len(fgraph.clients[mm]) != 1:
        return None
    consumer, _ = fgraph.clients[mm][0]
    if consumer == "output":
        return None

    one = pt.constant(np.asarray(1.0, mm.type.dtype))
    alpha, mm_term = one, mm
    # optional scalar scaling sitting between the matmul and the add: alpha * (A@B)
    if consumer.op == pt.mul:
        s, rest = _peel_scalar(consumer.outputs[0])
        if s is None or rest is not mm:
            return None
        alpha, mm_term = s, consumer.outputs[0]
        if len(fgraph.clients[mm_term]) != 1:
            return None
        consumer, _ = fgraph.clients[mm_term][0]
        if consumer == "output":
            return None
    if consumer.op != pt.add:
        return None

    add_node = consumer
    others = [i for i in add_node.inputs if i is not mm_term]
    if not others:
        return None
    C = (
        others[0] if len(others) == 1 else pt.add(*others)
    )  # C may broadcast; no rank-lift needed
    s, rest = _peel_scalar(C)  # look up beta off the accumulator
    beta, C = (s, rest) if s is not None else (one, C)
    return {
        add_node.outputs[0]: canonical_gemm(
            A, B, C, alpha.astype(mm.type.dtype), beta.astype(mm.type.dtype)
        )
    }


BACKENDS = {
    "cvm": Mode(linker="cvm", optimizer="fast_run"),
    "numba": "NUMBA",
    "jax": "JAX",
}


def main():
    rng = np.random.default_rng(0)
    N, din, h = 64, 5, 8
    W, V = rng.standard_normal((din, h)), rng.standard_normal((din, h))
    bvec, brow, bmat = (
        rng.standard_normal(h),
        rng.standard_normal((1, h)),
        rng.standard_normal((N, h)),
    )
    x = rng.standard_normal((N, din))

    cases = [
        ("@ + vector bias (h,)", lambda X: X @ W + bvec, lambda x: x @ W + bvec),
        ("@ + row bias (1,h)", lambda X: X @ W + brow, lambda x: x @ W + brow),
        ("@ + matrix bias (N,h)", lambda X: X @ W + bmat, lambda x: x @ W + bmat),
        ("pt.dot + vector bias", lambda X: pt.dot(X, W) + bvec, lambda x: x @ W + bvec),
        (
            "2.5*(@) + bias  [alpha]",
            lambda X: 2.5 * (X @ W) + bvec,
            lambda x: 2.5 * (x @ W) + bvec,
        ),
        (
            "(@) + 3.0*bias  [beta]",
            lambda X: X @ W + 3.0 * bvec,
            lambda x: x @ W + 3.0 * bvec,
        ),
        (
            "2.0*(@) + 3.0*bias [a,b]",
            lambda X: 2.0 * (X @ W) + 3.0 * bvec,
            lambda x: 2.0 * (x @ W) + 3.0 * bvec,
        ),
        ("residual @ + @", lambda X: X @ W + X @ V, lambda x: x @ W + x @ V),
        ("bare @ (no bias)", lambda X: X @ W, lambda x: x @ W),
    ]
    print("fwd: gemm=1 means CanonicalGEMM in graph, ok=1 means matches naive ref\n")
    for label, build, ref in cases:
        X = pt.matrix("X", dtype="float64")
        out = build(X)
        ref_val = ref(x)
        row = [f"{label:26s}"]
        for name, mode in BACKENDS.items():
            f = pytensor.function([X], out, mode=mode)
            has = "CanonicalGEMM" in _dprint(f)
            ok = np.allclose(np.asarray(f(x)), ref_val, atol=1e-9)
            row.append(f"{name}:gemm={int(has)},ok={int(ok)}")
        print("  ".join(row))

    # gradient check: differentiate an EXPLICIT CanonicalGEMM node (exercises L_op)
    print(
        "\nL_op check: grad of sum(CanonicalGEMM(X, W, b, 1, 1)) vs reference, across backends:"
    )
    Wc = pt.constant(W)
    one = pt.constant(np.asarray(1.0))
    ref_gX = np.ones((N, h)) @ W.T  # dX of sum(X@W + b)
    ref_gb = np.full(h, N, dtype=float)  # db: each b_j enters N rows
    for name, mode in BACKENDS.items():
        X = pt.matrix("X", dtype="float64")
        b = pt.vector("b", dtype="float64")
        y = canonical_gemm(X, Wc, b, one, one).sum()
        gX, gb = pt.grad(y, [X, b])
        grad_graph = _dprint(pytensor.function([X, b], [gX, gb], mode=mode))
        gXv, gbv = pytensor.function([X, b], [gX, gb], mode=mode)(x, bvec)
        ok = np.allclose(gXv, ref_gX, atol=1e-9) and np.allclose(gbv, ref_gb, atol=1e-9)
        grad_has_cg = "CanonicalGEMM" in grad_graph
        print(
            f"  {name}: grad_ok={int(ok)}  grad_reuses_CanonicalGEMM={int(grad_has_cg)}"
        )

    # static-shape propagation through make_node
    print("\nstatic shape + in-place:")
    As = pt.tensor("A", shape=(N, din), dtype="float64")
    node_out = canonical_gemm(
        As, pt.constant(W), pt.zeros(()), pt.constant(1.0), pt.constant(1.0)
    )
    print(f"  out.type.shape = {node_out.type.shape}  (expected ({N}, {h}))")

    # inplace_on_inputs only enables overwrite_c when C (input 2) is offered
    print(
        f"  inplace_on_inputs([2]).overwrite_c   = {canonical_gemm.inplace_on_inputs([2]).overwrite_c}  (expect True)"
    )
    print(
        f"  inplace_on_inputs([0,1]).overwrite_c = {canonical_gemm.inplace_on_inputs([0, 1]).overwrite_c}  (expect False)"
    )

    # the overwrite_c perform genuinely destroys C's buffer
    op_ip = CanonicalGEMM(overwrite_c=True)
    Av, Bv = rng.standard_normal((4, 3)), rng.standard_normal((3, 4))
    Cv = np.asfortranarray(
        rng.standard_normal((4, 4))
    )  # F-contiguous so scipy gemm can overwrite in place
    C_before, C_obj = Cv.copy(), Cv
    node = op_ip.make_node(
        pt.as_tensor_variable(Av),
        pt.as_tensor_variable(Bv),
        pt.as_tensor_variable(Cv),
        pt.constant(2.0),
        pt.constant(3.0),
    )
    out_storage = [[None]]
    op_ip.perform(node, [Av, Bv, Cv, np.float64(2.0), np.float64(3.0)], out_storage)
    result = out_storage[0][0]
    reused = result is C_obj
    destroyed = not np.allclose(C_obj, C_before)
    correct = np.allclose(result, 2.0 * (Av @ Bv) + 3.0 * C_before)
    print(
        f"  overwrite_c perform: reuses_C_buffer={int(reused)}  destroyed_C={int(destroyed)}  correct={int(correct)}"
    )


import io
import contextlib


def _dprint(obj):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pytensor.dprint(obj)
    return buf.getvalue()


if __name__ == "__main__":
    main()
