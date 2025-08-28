import numpy as np
from numba import njit
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


def nb_solve_continuous_lyapunov(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    return linalg.solve_continuous_lyapunov(a=a, q=q)


@overload(nb_solve_continuous_lyapunov)
def solve_continuous_lyapunov_impl(A, Q):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "solve_continuous_lyapunov")
    _check_scipy_linalg_matrix(Q, "solve_continuous_lyapunov")

    dtype = A.dtype
    w_type = _get_underlying_float(dtype)

    numba_xtrsyl = _LAPACK().numba_xtrsyl(dtype)

    def _solve_cont_lyapunov_impl(A, Q):
        _M, _N = np.int32(A.shape)
        _NQ = np.int32(Q.shape[-1])

        is_complex = np.iscomplexobj(A) | np.iscomplexobj(Q)
        dtype_letter = "C" if is_complex else "T"

        if A.flags.f_contiguous:
            A_copy = A
        else:
            A_copy = _copy_to_fortran_order(A)

        if Q.flags.f_configuous:
            Q_copy = Q
        else:
            Q_copy = _copy_to_fortran_order(Q)

        R, U = linalg.schur(A_copy, overwrite_a=True)

        # Construct f = u'*q*u
        F = U.conj().T.dot(Q_copy.dot(U))

        TRANA = val_to_int_ptr(ord("N"))
        TRANB = val_to_int_ptr(ord(dtype_letter))
        ISGN = val_to_int_ptr(1)

        M = val_to_int_ptr(_N)
        N = val_to_int_ptr(_N)
        LDA = val_to_int_ptr(_N)
        LDB = val_to_int_ptr(_N)
        LDC = val_to_int_ptr(_N)

        SCALE = np.array(1.0, dtype=w_type)
        INFO = val_to_int_ptr(1)

        numba_xtrsyl(
            TRANA,
            TRANB,
            ISGN,
            M,
            N,
            R.view(w_type).ctypes,
            LDA,
            R.view(w_type).ctypes,
            LDB,
            F.view(w_type).ctypes,
            LDC,
            SCALE.ctypes,
            INFO,
        )

        F *= SCALE
        _handle_err_maybe_convergence_problem(int_ptr_to_val(INFO))
        X = U.dot(F).dot(U.conj().T)

        return X

    return _solve_cont_lyapunov_impl


@njit
def direct_lyapunov_solution(A, B):
    lhs = np.kron(A, A.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = np.linalg.solve(lhs, B.flatten())

    return np.reshape(x, B.shape)


def nb_solve_discrete_lyapunov(a, q, method="auto"):
    return linalg.solve_discrete_lyapunov(a=a, q=q, method=method)


@overload(nb_solve_discrete_lyapunov)
def solve_discrete_lyapunov_impl(A, Q, method="auto"):
    ensure_lapack()

    _check_scipy_linalg_matrix(A, "solve_continuous_lyapunov")
    _check_scipy_linalg_matrix(Q, "solve_continuous_lyapunov")

    # dtype = A.dtype
    # w_type = _get_underlying_float(dtype)  #  noeq

    def impl(A, Q, method="auto"):
        _M, _N = np.int32(A.shape)

        if method == "auto":
            if _M < 10:
                method = "direct"
            else:
                method = "bilinear"

        if method == "direct":
            X = direct_lyapunov_solution(A, Q)

        if method == "bilinear":
            eye = np.eye(_M)
            AH = A.conj().transpose()
            AHI_inv = np.linalg.inv(AH + eye)
            B = np.dot(AH - eye, AHI_inv)
            C = 2 * np.dot(np.dot(np.linalg.inv(A + eye), Q), AHI_inv)
            X = linalg.solve_continuous_lyapunov(B.conj().transpose(), -C)

        return X

    return impl
