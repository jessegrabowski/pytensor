import typing
from typing import TYPE_CHECKING

import numpy as np

from pytensor.graph import Apply, Op
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor, diagonal
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
