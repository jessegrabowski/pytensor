"""Header text for the C and Fortran BLAS interfaces.

There is no standard name or location for this header, so we just insert it
ourselves into the C code. The static C declarations live as Python strings in
:mod:`pytensor.tensor.blas._c_code`; this module assembles the complete header
text, adding dynamic parts like the macOS sdot bug workaround.
"""

import functools
import logging
import os
import sys

from pytensor.configdefaults import config
from pytensor.link.c.cmodule import GCC_compiler
from pytensor.tensor.blas._c_code import (
    ALT_BLAS_COMMON,
    ALT_BLAS_TEMPLATE,
    FORTRAN_BLAS,
    MACOS_SDOT_ERROR,
    MACOS_SDOT_FIX_TEST,
    MACOS_SDOT_TEST,
    MACOS_SDOT_WORKAROUND,
    MKL_THREADS,
    OPENBLAS_THREADS,
)


_logger = logging.getLogger("pytensor.tensor.blas")


def detect_macos_sdot_bug():
    """
    Try to detect a bug in the BLAS sdot_ routine on macOS.

    Apple's Accelerate framework has a long-standing bug where the Fortran
    interface sdot_() returns incorrect values. The C interface cblas_sdot()
    works correctly. This bug has been present since at least macOS 10.6
    and is STILL PRESENT as of macOS 26 (2026).

    This function compiles and runs a test program to detect the bug,
    then tests if a workaround (using cblas_sdot instead) works.

    Three attributes of this function will be set:
        - detect_macos_sdot_bug.tested: True after first call
        - detect_macos_sdot_bug.present: True if bug is detected
        - detect_macos_sdot_bug.fix_works: True if cblas_sdot workaround works
    """
    _logger.debug("Starting detection of bug in Mac OS BLAS sdot_ routine")
    if detect_macos_sdot_bug.tested:
        return detect_macos_sdot_bug.present

    if sys.platform != "darwin" or not config.blas__ldflags:
        _logger.info("Not Mac OS, no sdot_ bug")
        detect_macos_sdot_bug.tested = True
        return False

    # This code will return -1 if the dot product did not return
    # the right value (30.).
    flags = config.blas__ldflags.split()
    for f in flags:
        # Library directories should also be added as rpath,
        # so that they can be loaded even if the environment
        # variable LD_LIBRARY_PATH does not contain them
        lib_path = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":")
        if f.startswith("-L"):
            flags.append("-Wl,-rpath," + f[2:])
            # also append those paths to DYLD_FALLBACK_LIBRARY_PATH to
            # support libraries that have the wrong install_name
            # (such as MKL on canopy installs)
            if f[2:] not in lib_path:
                lib_path.append(f[2:])
        # this goes into the python process environment that is
        # inherited by subprocesses/used by dyld when loading new objects
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(lib_path)

    test_code = MACOS_SDOT_TEST

    _logger.debug("Trying to compile and run test case.")
    compilation_ok, run_ok = GCC_compiler.try_compile_tmp(
        test_code, tmp_prefix="detect_macos_sdot_bug_", flags=flags, try_run=True
    )
    detect_macos_sdot_bug.tested = True

    # If compilation failed, we consider there is a bug,
    # and the fix does not work
    if not compilation_ok:
        _logger.info("Could not compile test case for sdot_.")
        detect_macos_sdot_bug.present = True
        return True

    if run_ok:
        _logger.info("The sdot_ bug is not present on this system.")
        detect_macos_sdot_bug.present = False
        return False

    # Else, the bug is detected.
    _logger.info("The sdot_ bug is present on this system.")
    detect_macos_sdot_bug.present = True

    # Then, try a simple fix
    test_fix_code = MACOS_SDOT_FIX_TEST

    _logger.debug("Trying to compile and run tentative workaround.")
    compilation_fix_ok, run_fix_ok = GCC_compiler.try_compile_tmp(
        test_fix_code,
        tmp_prefix="detect_macos_sdot_bug_testfix_",
        flags=flags,
        try_run=True,
    )

    _logger.info(
        "Status of tentative fix -- compilation OK: %s, works: %s",
        compilation_fix_ok,
        run_fix_ok,
    )
    detect_macos_sdot_bug.fix_works = run_fix_ok

    return detect_macos_sdot_bug.present


detect_macos_sdot_bug.tested = False  # type: ignore[attr-defined]
detect_macos_sdot_bug.present = False  # type: ignore[attr-defined]
detect_macos_sdot_bug.fix_works = False  # type: ignore[attr-defined]


def blas_header_text():
    """C header for the fortran blas interface.

    Returns the complete BLAS header text including:
    - Fortran BLAS declarations
    - macOS sdot bug workaround (if applicable)
    - NumPy-based fallback BLAS (if no system BLAS available)
    """
    blas_code = ""
    if not config.blas__ldflags:
        # This code can only be reached by compiling a function with a manually specified GEMM Op.
        # Normal PyTensor usage will end up with Dot22 or Dot22Scalar instead,
        # which opt out of C-code completely if the blas flags are missing
        _logger.warning("Using NumPy C-API based implementation for BLAS functions.")

        # Include the Numpy version implementation of [sd]gemm_.
        sblas_code = ALT_BLAS_TEMPLATE % {
            "float_type": "float",
            "float_size": 4,
            "npy_float": "NPY_FLOAT32",
            "precision": "s",
        }
        dblas_code = ALT_BLAS_TEMPLATE % {
            "float_type": "double",
            "float_size": 8,
            "npy_float": "NPY_FLOAT64",
            "precision": "d",
        }
        blas_code += ALT_BLAS_COMMON
        blas_code += sblas_code
        blas_code += dblas_code

    header = FORTRAN_BLAS

    # Add macOS sdot bug workaround if needed
    if detect_macos_sdot_bug():
        if detect_macos_sdot_bug.fix_works:
            header += MACOS_SDOT_WORKAROUND
        else:
            # Make sure the buggy version of sdot_ is never used
            header += MACOS_SDOT_ERROR

    return header + blas_code


@functools.cache
def mkl_threads_text():
    """C header for MKL threads interface."""
    return MKL_THREADS


@functools.cache
def openblas_threads_text():
    """C header for OpenBLAS threads interface."""
    return OPENBLAS_THREADS


def blas_header_version():
    """Return version tuple for cache invalidation.

    This version should be bumped when:
    - The static BLAS C source changes
    - The sdot bug workaround logic changes
    """
    # Version 14: BLAS C source inlined as Python strings (no on-disk .c/.h)
    version = (14,)
    if detect_macos_sdot_bug():
        if detect_macos_sdot_bug.fix_works:
            version += (1,)
        else:
            version += (2,)
    return version
