import typing
from typing import TYPE_CHECKING

import numpy as np

from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import (
    arange,
    as_tensor,
    concatenate,
    diagonal,
    ones_like,
    where,
    zeros_like,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg._lazy import scipy_linalg
from pytensor.tensor.type import tensor, vector
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class LUFactorTridiagonal(Op):
    """Compute LU factorization of a tridiagonal matrix (lapack gttrf)"""

    __props__ = (
        "overwrite_dl",
        "overwrite_d",
        "overwrite_du",
    )
    gufunc_signature = "(dl),(d),(dl)->(dl),(d),(dl),(du2),(d)"

    def __init__(self, overwrite_dl=False, overwrite_d=False, overwrite_du=False):
        self.destroy_map = dm = {}
        if overwrite_dl:
            dm[0] = [0]
        if overwrite_d:
            dm[1] = [1]
        if overwrite_du:
            dm[2] = [2]
        self.overwrite_dl = overwrite_dl
        self.overwrite_d = overwrite_d
        self.overwrite_du = overwrite_du
        super().__init__()

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        return type(self)(
            overwrite_dl=0 in allowed_inplace_inputs,
            overwrite_d=1 in allowed_inplace_inputs,
            overwrite_du=2 in allowed_inplace_inputs,
        )

    def make_node(self, dl, d, du):
        dl, d, du = map(as_tensor, (dl, d, du))

        if not all(inp.type.ndim == 1 for inp in (dl, d, du)):
            raise ValueError("Diagonals must be vectors")

        ndl, nd, ndu = (inp.type.shape[-1] for inp in (dl, d, du))

        # An off-diagonal of length 0 does not determine n, which can be 0 or 1
        match (ndl, nd, ndu):
            case (_, int(), _):
                n = nd
            case (int(), _, _) if ndl > 0:
                n = ndl + 1
            case (_, _, int()) if ndu > 0:
                n = ndu + 1
            case _:
                n = None

        dummy_arrays = [np.zeros((), dtype=inp.type.dtype) for inp in (dl, d, du)]
        out_dtype = scipy_linalg.get_lapack_funcs("gttrf", dummy_arrays).dtype
        n_off = None if n is None else max(n - 1, 0)
        n_fill = None if n is None else max(n - 2, 0)
        outputs = [
            vector(shape=(n_off,), dtype=out_dtype),
            vector(shape=(n,), dtype=out_dtype),
            vector(shape=(n_off,), dtype=out_dtype),
            vector(shape=(n_fill,), dtype=out_dtype),
            vector(shape=(n,), dtype=np.int32),
        ]
        return Apply(self, [dl, d, du], outputs)

    def infer_shape(self, node, shapes):
        dl_shape, d_shape, du_shape = shapes
        [n] = d_shape
        return [dl_shape, d_shape, du_shape, (ptm.maximum(n - 2, 0),), d_shape]

    def connection_pattern(self, node):
        return [[True, True, True, True, False] for _ in node.inputs]

    def perform(self, node, inputs, output_storage):
        dtype = node.outputs[0].type.dtype
        dl, d, du = inputs
        n = d.shape[0]
        overwrite_dl, overwrite_d, overwrite_du = (
            self.overwrite_dl,
            self.overwrite_d,
            self.overwrite_du,
        )

        # scipy's f2py wrapper infers the wrong size when n < 3. The factors of
        # diag(A, I) start with those of A.
        if n < 3:
            dl, du = (np.pad(v, (0, 2 - v.size)) for v in (dl, du))
            d = np.pad(d, (0, 3 - n), constant_values=1)
            overwrite_dl = overwrite_d = overwrite_du = True

        gttrf = scipy_linalg.get_lapack_funcs("gttrf", dtype=dtype)
        dl, d, du, du2, ipiv, _ = gttrf(
            dl,
            d,
            du,
            overwrite_dl=overwrite_dl,
            overwrite_d=overwrite_d,
            overwrite_du=overwrite_du,
        )

        output_storage[0][0] = dl[: max(n - 1, 0)]
        output_storage[1][0] = d[:n]
        output_storage[2][0] = du[: max(n - 1, 0)]
        output_storage[3][0] = du2[: max(n - 2, 0)]
        output_storage[4][0] = ipiv[:n]

    def pullback(self, inputs, outputs, output_grads):
        """Reverse-mode gradient of gttrf, holding the pivot sequence fixed.

        Reversing the elimination carries the cotangents of the running pivot
        ``d[i + 1]`` and of the fill-in slot ``du[i + 1]`` from step i back to step
        i - 1. Eliminating the fill-in carry leaves a second order linear recurrence in
        the pivot carry, which is a unit upper triangular system with two
        superdiagonals.
        """
        dl, d, du, du2, ipiv = outputs
        dl_bar, d_bar, du_bar, du2_bar = (
            zeros_like(output) if isinstance(grad.type, DisconnectedType) else grad
            for output, grad in zip(outputs[:4], output_grads[:4], strict=True)
        )

        n = d.shape[0]
        swap = ptm.neq(ipiv[:-1], arange(1, n))
        pivots = d[:-1]

        # The fill-in carry entering step i is
        # fill_scale[i + 1] * pivot_carry[i + 2] + fill_offset[i + 1]
        fill_scale = where(swap, 1, -dl)
        fill_offset = where(swap, 0, du_bar)

        carry_du = where(swap, du / pivots, -du * dl / pivots)
        carry_du2 = where(swap[:-1], du2 * fill_scale[1:] / pivots[:-1], 0)
        carry_rhs = where(
            swap,
            dl_bar[:-1].inc(-du2 * fill_offset[1:]) / pivots,
            d_bar[:-1] - dl_bar * dl / pivots,
        )
        pivot_carry = SolveLUFactorTridiagonal(b_ndim=1, transposed=False)(
            zeros_like(dl),
            ones_like(d),
            carry_du,
            carry_du2,
            arange(1, n + 1, dtype="int32"),
            concatenate([carry_rhs, d_bar[-1:]]),
        )

        next_pivot_carry = pivot_carry[1:]
        fill_carry = fill_scale * next_pivot_carry + fill_offset

        multiplier_bar = (dl_bar - next_pivot_carry * du)[:-1].inc(
            -where(swap[:-1], fill_carry[1:] * du2, 0)
        )
        dl_in_bar = where(
            swap,
            d_bar[:-1] - multiplier_bar * dl / pivots,
            multiplier_bar / pivots,
        )
        d_in_bar = concatenate(
            [
                pivot_carry[:1],
                where(swap, du_bar - next_pivot_carry * dl, next_pivot_carry),
            ]
        )
        du_in_bar = concatenate(
            [
                fill_carry[:1],
                where(swap[:-1], du2_bar - fill_carry[1:] * dl[:-1], fill_carry[1:]),
            ]
        )
        return [dl_in_bar, d_in_bar, du_in_bar]


class SolveLUFactorTridiagonal(Op):
    """Solve a system of linear equations with a tridiagonal coefficient matrix (lapack gttrs)."""

    __props__ = ("b_ndim", "overwrite_b", "transposed")

    def __init__(self, b_ndim: int, transposed: bool, overwrite_b=False):
        if b_ndim not in (1, 2):
            raise ValueError("b_ndim must be 1 or 2")
        if b_ndim == 1:
            self.gufunc_signature = "(dl),(d),(dl),(du2),(d),(d)->(d)"
        else:
            self.gufunc_signature = "(dl),(d),(dl),(du2),(d),(d,rhs)->(d,rhs)"
        if overwrite_b:
            self.destroy_map = {0: [5]}
        self.b_ndim = b_ndim
        self.transposed = transposed
        self.overwrite_b = overwrite_b
        super().__init__()

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        # b matrix is the 5th input
        if 5 in allowed_inplace_inputs:
            props = self._props_dict()  # type: ignore
            props["overwrite_b"] = True
            return type(self)(**props)

        return self

    def make_node(self, dl, d, du, du2, ipiv, b):
        dl, d, du, du2, ipiv, b = map(as_tensor, (dl, d, du, du2, ipiv, b))

        if b.type.ndim != self.b_ndim:
            raise ValueError("Wrong number of dimensions for input b.")

        if not all(inp.type.ndim == 1 for inp in (dl, d, du, du2, ipiv)):
            raise ValueError("Inputs must be vectors")

        ndl, nd, ndu, nipiv = (inp.type.shape[-1] for inp in (dl, d, du, ipiv))
        nb = b.type.shape[0]

        # du2 has length max(n - 2, 0), and an off-diagonal of length 0 allows n of 0
        # or 1, so neither determines n
        match (ndl, nd, ndu, nipiv):
            case (_, int(), _, _):
                n = nd
            case (_, _, _, int()):
                n = nipiv
            case (int(), _, _, _) if ndl > 0:
                n = ndl + 1
            case (_, _, int(), _) if ndu > 0:
                n = ndu + 1
            case _:
                n = nb

        dummy_arrays = [
            np.zeros((), dtype=inp.type.dtype) for inp in (dl, d, du, du2, b)
        ]
        out_dtype = scipy_linalg.get_lapack_funcs("gttrs", dummy_arrays).dtype
        if self.b_ndim == 1:
            output_shape = (n,)
        else:
            output_shape = (n, b.type.shape[-1])

        outputs = [tensor(shape=output_shape, dtype=out_dtype)]
        return Apply(self, [dl, d, du, du2, ipiv, b], outputs)

    def infer_shape(self, node, shapes):
        return [shapes[5]]

    def connection_pattern(self, node):
        return [[True], [True], [True], [True], [False], [True]]

    def perform(self, node, inputs, output_storage):
        dtype = node.outputs[0].type.dtype
        dl, d, du, du2, ipiv, b = inputs
        n = d.shape[0]
        overwrite_b = self.overwrite_b

        # scipy's f2py wrapper infers the wrong size when n < 3. Solve with the
        # factors of diag(A, I) instead.
        if n < 3:
            dl, du = (np.pad(v, (0, 2 - v.size)) for v in (dl, du))
            d = np.pad(d, (0, 3 - n), constant_values=1)
            du2 = np.pad(du2, (0, 1 - du2.size))
            ipiv = np.concatenate([ipiv, np.arange(n + 1, 4, dtype=ipiv.dtype)])
            b = np.pad(b, [(0, 3 - n)] + [(0, 0)] * (b.ndim - 1))
            overwrite_b = True

        gttrs = scipy_linalg.get_lapack_funcs("gttrs", dtype=dtype)
        x, _ = gttrs(
            dl,
            d,
            du,
            du2,
            ipiv,
            b,
            overwrite_b=overwrite_b,
            trans="N" if not self.transposed else "T",
        )
        output_storage[0][0] = x[:n]

    def pullback(self, inputs, outputs, output_grads):
        """Reverse-mode gradient of gttrs.

        With ``A = P L U``, gttrs runs an L stage of pivoted row operations with
        multipliers ``dl`` and a banded U stage. The U cotangents are band products of
        the U stage solution and the cotangent of its right-hand side. The ``dl``
        cotangents depend on the values carried between rows in the L stage and on
        their cotangents. Each carry is a first order linear recurrence, which is a
        bidiagonal solve.
        """
        dl, d, du, du2, ipiv, b = inputs
        [x] = outputs
        [x_bar] = output_grads

        b_bar = type(self)(b_ndim=self.b_ndim, transposed=not self.transposed)(
            dl, d, du, du2, ipiv, x_bar
        )
        b_bar_out = b_bar
        if self.b_ndim == 1:
            b, x, x_bar, b_bar = (v[:, None] for v in (b, x, x_bar, b_bar))

        n = d.shape[0]
        swap = ptm.neq(ipiv[:-1], arange(1, n))[:, None]

        # With no multipliers and no row swaps, gttrs only runs its banded U stage
        no_dl = zeros_like(dl)
        no_swaps = arange(1, n + 1, dtype="int32")
        u_solve = SolveLUFactorTridiagonal(b_ndim=2, transposed=False)
        ut_solve = SolveLUFactorTridiagonal(b_ndim=2, transposed=True)
        carry_factors = (
            no_dl,
            ones_like(d),
            where(swap[:, 0], -1, dl),
            zeros_like(du2),
            no_swaps,
        )

        if not self.transposed:
            # x = U^-1 y, where y is the output of the L stage
            y_bar = ut_solve(no_dl, d, du, du2, no_swaps, x_bar)
            y = (d[:, None] * x)[:-1].inc(du[:, None] * x[1:])
            y = y[:-2].inc(du2[:, None] * x[2:])

            carry_bar = u_solve(
                *carry_factors, y_bar[:-1].set(where(swap, 0, y_bar[:-1]))
            )
            dl_bar = (carry_bar[1:] * y[:-1]).sum(-1)
            outer_left, outer_right = y_bar, x
        else:
            # z = U^-T b is the input of the L stage
            z = ut_solve(no_dl, d, du, du2, no_swaps, b)

            carry = u_solve(*carry_factors, z[:-1].set(where(swap, 0, z[:-1])))
            carry_bar = ut_solve(
                *carry_factors,
                x_bar[1:].set(where(swap, -dl[:, None] * x_bar[1:], x_bar[1:])),
            )
            dl_bar = (where(swap, x_bar[1:], carry_bar[:-1]) * carry[1:]).sum(-1)
            outer_left, outer_right = z, b_bar

        # The U cotangent is -outer_left @ outer_right.T, restricted to the band of U
        d_bar = (outer_left * outer_right).sum(-1)
        du_bar = (outer_left[:-1] * outer_right[1:]).sum(-1)
        du2_bar = (outer_left[:-2] * outer_right[2:]).sum(-1)

        return [-dl_bar, -d_bar, -du_bar, -du2_bar, DisconnectedType()(), b_bar_out]


def tridiagonal_lu_factor(
    a: "TensorLike",
) -> tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
]:
    """Return the decomposition of A implied by a solve tridiagonal (LAPACK's gttrf)

    Parameters
    ----------
    a
        The input matrix.

    Returns
    -------
    dl, d, du, du2, ipiv
        The LU factorization of A.
    """
    dl, d, du = (diagonal(a, offset=o, axis1=-2, axis2=-1) for o in (-1, 0, 1))
    dl, d, du, du2, ipiv = typing.cast(
        list[TensorVariable], Blockwise(LUFactorTridiagonal())(dl, d, du)
    )
    return dl, d, du, du2, ipiv


def tridiagonal_lu_solve(
    a_diagonals: tuple[
        "TensorLike", "TensorLike", "TensorLike", "TensorLike", "TensorLike"
    ],
    b: "TensorLike",
    *,
    b_ndim: int,
    transposed: bool = False,
) -> TensorVariable:
    """Solve a tridiagonal system of equations using LU factorized inputs (LAPACK's gttrs).

    Parameters
    ----------
    a_diagonals
        The outputs of tridiagonal_lu_factor(A).
    b
        The right-hand side vector or matrix.
    b_ndim
        The number of dimensions of the right-hand side.
    transposed
        Whether to solve the transposed system.

    Returns
    -------
    TensorVariable
        The solution vector or matrix.
    """
    dl, d, du, du2, ipiv = a_diagonals
    return typing.cast(
        TensorVariable,
        Blockwise(SolveLUFactorTridiagonal(b_ndim=b_ndim, transposed=transposed))(
            dl, d, du, du2, ipiv, b
        ),
    )
