from typing import Any, Literal

import numpy as np
from numba.core import types
from numba.core.extending import overload
from numba.np.linalg import (
    _copy_to_fortran_order,
    _handle_err_maybe_convergence_problem,
    ensure_lapack,
)
from scipy import linalg

from pytensor.link.numba.dispatch.linalg._LAPACK import (
    _LAPACK,
    _get_underlying_float,
    int_ptr_to_val,
    val_to_int_ptr,
)
from pytensor.link.numba.dispatch.linalg.utils import _check_scipy_linalg_matrix


def _schur(
    a: Any,
    output: Literal["real", "r"] = "real",
    lwork: Any | None = None,
    overwrite_a: Any = False,
    sort: None = None,
    check_finite: Any = True,
) -> tuple[Any, Any]:
    return linalg.schur(
        a,
        output=output,
        lwork=lwork,
        overwrite_a=overwrite_a,
        sort=sort,
        check_finite=check_finite,
    )


@overload(_schur)
def schur_impl(A, output, lwork, overwrite_a, sort, check_finite):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "schur")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    numba_rgees = _LAPACK().numba_rgees(dtype)
    numba_cgees = _LAPACK().numba_cgees(dtype)

    def real_schur_impl(A, output, lwork, overwrite_a, sort, check_finite):
        """
        schur() implementation for real arrays
        """
        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            msg = "Last 2 dimensions of the array must be square"
            raise linalg.LinAlgError(msg)

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        JOBVS = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0.0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(_N)
        WR = np.empty(_N, dtype=dtype)
        WI = np.empty(_N, dtype=dtype)
        _LDVS = _N
        LDVS = val_to_int_ptr(_N)
        VS = np.empty((_LDVS, _N), dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        WORK = np.empty(1, dtype=dtype)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_rgees(
            JOBVS,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            SDIM,
            WR.ctypes,
            WI.ctypes,
            VS.ctypes,
            LDVS,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )
        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual work
        numba_rgees(
            JOBVS,
            SORT,
            SELECT,
            N,
            A_copy.ctypes,
            LDA,
            SDIM,
            WR.ctypes,
            WI.ctypes,
            VS.ctypes,
            LDVS,
            WORK.ctypes,
            LWORK,
            BWORK,
            INFO,
        )

        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))

        return A_copy, VS.T

    def complex_schur_impl(A, output, lwork, overwrite_a, sort, check_finite):
        """
        schur() implementation for complex arrays
        """

        _N = np.int32(A.shape[-1])
        if A.shape[-2] != _N:
            msg = "Last 2 dimensions of the array must be square"
            raise linalg.LinAlgError(msg)

        if overwrite_a and A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        JOBVS = val_to_int_ptr(ord("V"))
        SORT = val_to_int_ptr(ord("N"))
        SELECT = val_to_int_ptr(0.0)

        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        SDIM = val_to_int_ptr(_N)
        W = np.empty(_N, dtype=dtype)
        _LDVS = _N
        LDVS = val_to_int_ptr(_N)
        VS = np.empty((_LDVS, _N), dtype=dtype)
        LWORK = val_to_int_ptr(-1)
        WORK = np.empty(1, dtype=dtype)
        RWORK = np.empty(_N, dtype=w_type)
        BWORK = val_to_int_ptr(1)
        INFO = val_to_int_ptr(1)

        # workspace query
        numba_cgees(
            JOBVS,
            SORT,
            SELECT,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            SDIM,
            W.view(w_type).ctypes,
            VS.view(w_type).ctypes,
            LDVS,
            WORK.view(w_type).ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        WS_SIZE = np.int32(WORK[0].real)
        LWORK = val_to_int_ptr(WS_SIZE)
        WORK = np.empty(WS_SIZE, dtype=dtype)

        # Actual work
        numba_cgees(
            JOBVS,
            SORT,
            SELECT,
            N,
            A_copy.view(w_type).ctypes,
            LDA,
            SDIM,
            W.view(w_type).ctypes,
            VS.view(w_type).ctypes,
            LDVS,
            WORK.view(w_type).ctypes,
            LWORK,
            RWORK.ctypes,
            BWORK,
            INFO,
        )

        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))

        return A_copy, VS.T

    if isinstance(A.dtype, types.scalars.Complex):
        return complex_schur_impl
    else:
        return real_schur_impl
