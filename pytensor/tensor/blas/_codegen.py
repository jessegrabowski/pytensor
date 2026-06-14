# C-code generators for the BLAS ops, kept out of the Op classes so the Op
# files stay small and the generators can be reused by the C dispatch registry
# (``c_funcify``) as well as by the COp ``c_code`` methods. Each generator takes
# the Apply ``node`` plus the C variable names and returns a C source string,
# matching the ``c_code(node, name, inputs, outputs, sub)`` contract.
#
# The kernels are SPECIALIZED on the node's static dtype: a node is known at
# graph-construction time to be float32 or float64, so we emit only that
# precision's BLAS call instead of a runtime ``switch(type_num)``.
#
# !!! CACHE DISCIPLINE !!!
# The CLinker module cache does NOT hash the generated C text (see
# ``cmodule_key_`` in pytensor/link/c/basic.py). A node's cache key is built
# from ``c_code_cache_version_apply`` + ``__props__`` + the input/output type
# versions. dtype and static shape ride the type signature automatically, but
# ANY change to the emitted C must be reflected by BUMPING the leading version
# int in the corresponding op's ``c_code_cache_version`` -- otherwise callers
# silently get stale compiled binaries.

from collections.abc import Sequence
from enum import Enum, auto

from pytensor.graph.utils import MethodNotDefined


# Map a (statically known) float dtype to its C type, BLAS precision prefix,
# and element size in bytes.
_DTYPE_TO_C = {
    "float32": ("float", "s", 4),
    "float64": ("double", "d", 8),
}


class CODE_TOKEN(Enum):
    INDENT = auto()
    DEDENT = auto()
    EMPTY_LINE = auto()


def build_source_code(code: Sequence[str | CODE_TOKEN]) -> str:
    """Assemble C source from a token stream with managed indentation."""
    lines = []
    indentation_level = 0
    for line in code:
        if line is CODE_TOKEN.INDENT:
            indentation_level += 1
        elif line is CODE_TOKEN.DEDENT:
            indentation_level -= 1
            assert indentation_level >= 0
        elif line is CODE_TOKEN.EMPTY_LINE:
            lines.append("")
        else:
            lines.append(f"{'    ' * indentation_level}{line}")
    return "\n".join(lines)


# ##### ####### #######
# GEMM family (Gemm, Dot22, Dot22Scalar)
# ##### ####### #######

# Shared, dtype-independent GemmRelated template fragments, substituted with the
# ``%(name)s`` mechanism against the input/output C variable names.

_DECLARE_NS = """
        int unit = 0;

        int type_size = PyArray_ITEMSIZE(%(_x)s); // in bytes

        npy_intp* Nx = PyArray_DIMS(%(_x)s);
        npy_intp* Ny = PyArray_DIMS(%(_y)s);
        npy_intp* Nz = 0; //PyArray_DIMS(%(_zout)s);

        npy_intp* Sx = PyArray_STRIDES(%(_x)s);
        npy_intp* Sy = PyArray_STRIDES(%(_y)s);
        npy_intp* Sz = 0; //PyArray_STRIDES(%(_zout)s);

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;
        """

_CHECK_XYZ_RANK2 = """
        if (PyArray_NDIM(%(_x)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(x) != 2. rank(x) is %%d.",
                         PyArray_NDIM(%(_x)s));
            %(fail)s;
        }
        if (PyArray_NDIM(%(_y)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(y) != 2. rank(y) is %%d.", PyArray_NDIM(%(_y)s));
            %(fail)s;
        }
        if (%(_zout)s && PyArray_NDIM(%(_zout)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(z) != 2. rank(z) is %%d.", PyArray_NDIM(%(_zout)s));
            %(fail)s;
        }
        """

_CHECK_DIMS = """
        if (Nx[0] !=1 && Nz[0] != 1 && Nx[0] != Nz[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld rows but z has %%ld rows",
                (long int)Nx[0], (long int)Nz[0]);
            %(fail)s;
        }
        if (Nx[1] != Ny[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld cols (and %%ld rows) but y has %%ld rows (and %%ld cols)",
                (long int)Nx[1], (long int)Nx[0], (long int)Ny[0], (long int)Ny[1]);
            %(fail)s;
        }
        if (Ny[1] != 1 && Nz[1]!= 1 && Ny[1] != Nz[1])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: y has %%ld cols but z has %%ld cols",
                (long int)Ny[1], (long int)Nz[1]);
            %(fail)s;
        }

        // We must not raise an error when Nx[1] == 0. This would disable cases
        // that numpy.dot accept.
        """

_CHECK_STRIDES = """
        /*
        If some matrices are not contiguous on either dimensions,
        or have invalid strides, copy their content into a contiguous one
        */
        if (pytensor_needs_copy_for_blas(Nx, Sx, type_size))
        {
            PyArrayObject * _x_copy = (PyArrayObject *) PyArray_Copy(%(_x)s);
            if (!_x_copy)
                %(fail)s
            Py_XDECREF(%(_x)s);
            %(_x)s = _x_copy;
            Sx = PyArray_STRIDES(%(_x)s);
            if ((Sx[0] < 1) || (Sx[1] < 1)) {
                compute_strides(Nx, 2, type_size, Sx);
            }
        }

        if (pytensor_needs_copy_for_blas(Ny, Sy, type_size))
        {
            PyArrayObject * _y_copy = (PyArrayObject *) PyArray_Copy(%(_y)s);
            if (!_y_copy)
                %(fail)s
            Py_XDECREF(%(_y)s);
            %(_y)s = _y_copy;
            Sy = PyArray_STRIDES(%(_y)s);
            if ((Sy[0] < 1) || (Sy[1] < 1)) {
                compute_strides(Ny, 2, type_size, Sy);
            }
        }

        if (pytensor_needs_copy_for_blas(Nz, Sz, type_size))
        {
            PyArrayObject * _z_copy = (PyArrayObject *) PyArray_Copy(%(_zout)s);
            if (!_z_copy)
                %(fail)s
            Py_XDECREF(%(_zout)s);
            %(_zout)s = _z_copy;
            Sz = PyArray_STRIDES(%(_zout)s);
            if ((Sz[0] < 1) || (Sz[1] < 1)) {
                compute_strides(Nz, 2, type_size, Sz);
            }
        }
        """

_ENCODE_STRIDES_IN_UNIT = """
        /*
        encode the stride structure of _x,_y,_zout into a single integer
        */
        unit = pytensor_encode_gemm_strides(Nx, Sx, Ny, Sy, Nz, Sz, type_size);
        """

_COMPUTE_STRIDES = """
        /* create appropriate strides for malformed matrices that are row or column
         * vectors, or empty matrices.
         * In that case, the value of the stride does not really matter, but
         * some versions of BLAS insist that:
         *  - they are not smaller than the number of elements in the array,
         *  - they are not 0.
         */
        pytensor_compute_gemm_strides(Nx, Sx, &sx_0, &sx_1,
                                      Ny, Sy, &sy_0, &sy_1,
                                      Nz, Sz, &sz_0, &sz_1,
                                      type_size);
        """

# Gemm output-setup fragments (in-place broadcasts z onto the destroyed input;
# out-of-place allocates a fresh z and copies the input in).

_GEMM_SETUP_Z_INPLACE = """
        // Needs broadcasting
        if (PyArray_DIMS(%(_z)s)[0] < Nx[0] || PyArray_DIMS(%(_z)s)[1] < Ny[1]){

            npy_intp dims[2];
            dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
            dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

            // Check if we need to allocate new array
            if((NULL == %(_zout)s)
                || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
                || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
            {
                // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
                Py_XDECREF(%(_zout)s);
                %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            }

            // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
            if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
            {
                %(fail)s;
            }

        } else {
            if (%(_zout)s != %(_z)s)
            {
                Py_XDECREF(%(_zout)s);
                %(_zout)s = %(_z)s;
                Py_INCREF(%(_zout)s);
            }
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

_GEMM_SETUP_Z_OUTPLACE = """
        npy_intp dims[2];
        dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
        dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

        // Check if we need to allocate new array
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
            || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_no_inplace output");
                %(fail)s
            }
        }

        // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
        if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
        {
            %(fail)s
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

_GEMM_BROADCAST_XY = """
        // Broadcast X if needed
        if (Nz[0] > Nx[0])
        {
            npy_intp dims[2];
            dims[0] = Nz[0];
            dims[1] = Nx[1];
            // fprintf(stderr, "Gemm Broadcasting X into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *x_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!x_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_CopyInto(x_new, %(_x)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_x)s);
            %(_x)s = x_new;

            Nx = PyArray_DIMS(%(_x)s);
            Sx = PyArray_STRIDES(%(_x)s);
        }

        // Broadcast Y if needed
        if (Nz[1] > Ny[1])
        {
            npy_intp dims[2];
            dims[0] = Ny[0];
            dims[1] = Nz[1];
            // fprintf(stderr, "Gemm Broadcasting Y into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *y_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!y_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_CopyInto(y_new, %(_y)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_y)s);
            %(_y)s = y_new;

            Ny = PyArray_DIMS(%(_y)s);
            Sy = PyArray_STRIDES(%(_y)s);
        }

    """

_DOT22_SETUP_Z = """
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_x)s)[0])
            || (PyArray_DIMS(%(_zout)s)[1] != PyArray_DIMS(%(_y)s)[1]))
        {
            if (NULL != %(_zout)s) Py_XDECREF(%(_zout)s);
            npy_intp dims[2];
            dims[0] = PyArray_DIMS(%(_x)s)[0];
            dims[1] = PyArray_DIMS(%(_y)s)[1];
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims,
                            PyArray_TYPE(%(_x)s));
            //fprintf(stderr, "Dot Allocating %%i %%i\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc dot22 output");
                %(fail)s
            }
        }
        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);

        """


def _gemm_compute_block(ctype, prec, ab_constants):
    """The single-precision GEMM dispatch call, with ``a``/``b`` set by ``ab_constants``."""
    return f"""
        {{
            {ab_constants}
            {ctype}* x = ({ctype}*)PyArray_DATA(%(_x)s);
            {ctype}* y = ({ctype}*)PyArray_DATA(%(_y)s);
            {ctype}* z = ({ctype}*)PyArray_DATA(%(_zout)s);
            int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
            if (pytensor_{prec}gemm_dispatch(unit, x, y, z, a, b,
                                        Nz0, Nz1, Nx1,
                                        sx_0, sx_1, sy_0, sy_1, sz_0, sz_1) != 0) {{
                %(fail)s;
            }}
        }}
        """


def _check_blas_dtype(dtype, op_name):
    """Return ``(ctype, prec, elemsize)`` or fall back to ``perform`` via MethodNotDefined."""
    try:
        return _DTYPE_TO_C[dtype]
    except KeyError:
        raise MethodNotDefined(f"{op_name}.c_code: unsupported dtype {dtype}")


def gemm_c_code(node, name, inputs, outputs, sub):
    """C code for ``Gemm``: ``z <- b * z + a * dot(x, y)`` (in/out-of-place)."""
    _z, _a, _x, _y, _b = inputs
    (_zout,) = outputs
    ctype, prec, _ = _check_blas_dtype(node.inputs[0].type.dtype, "Gemm")
    # a and b share the matrices' dtype (enforced by Gemm.make_node).
    ab_constants = (
        f"{ctype} a = (({ctype}*)PyArray_DATA(%(_a)s))[0];\n"
        f"            {ctype} b = (({ctype}*)PyArray_DATA(%(_b)s))[0];"
    )
    # inplace is a static op prop, so emit only the relevant output setup.
    setup_z = _GEMM_SETUP_Z_INPLACE if node.op.inplace else _GEMM_SETUP_Z_OUTPLACE
    code = "".join(
        (
            _DECLARE_NS,
            _CHECK_XYZ_RANK2,
            setup_z,
            _GEMM_BROADCAST_XY,
            _CHECK_DIMS,
            _CHECK_STRIDES,
            _ENCODE_STRIDES_IN_UNIT,
            _COMPUTE_STRIDES,
            _gemm_compute_block(ctype, prec, ab_constants),
        )
    )
    return code % dict(_z=_z, _a=_a, _x=_x, _y=_y, _b=_b, _zout=_zout, **sub)


def dot22_c_code(node, name, inputs, outputs, sub):
    """C code for ``Dot22``: ``z <- dot(x, y)`` into a freshly allocated z."""
    _x, _y = inputs
    (_zout,) = outputs
    ctype, prec, _ = _check_blas_dtype(node.inputs[0].type.dtype, "Dot22")
    ab_constants = f"{ctype} a = 1.0;\n            {ctype} b = 0.0;"
    code = "".join(
        (
            _DECLARE_NS,
            _CHECK_XYZ_RANK2,
            _DOT22_SETUP_Z,
            _CHECK_DIMS,
            _CHECK_STRIDES,
            _ENCODE_STRIDES_IN_UNIT,
            _COMPUTE_STRIDES,
            _gemm_compute_block(ctype, prec, ab_constants),
        )
    )
    return code % dict(_x=_x, _y=_y, _zout=_zout, **sub)


def dot22scalar_c_code(node, name, inputs, outputs, sub):
    """C code for ``Dot22Scalar``: ``z <- a * dot(x, y)`` into a fresh z."""
    _x, _y, _a = inputs
    (_zout,) = outputs
    ctype, prec, _ = _check_blas_dtype(node.inputs[0].type.dtype, "Dot22Scalar")
    # a shares the matrices' dtype (enforced by Dot22Scalar.make_node).
    ab_constants = (
        f"{ctype} a = (({ctype}*)PyArray_DATA(%(_a)s))[0];\n"
        f"            {ctype} b = 0.0;"
    )
    code = "".join(
        (
            _DECLARE_NS,
            _CHECK_XYZ_RANK2,
            _DOT22_SETUP_Z,
            _CHECK_DIMS,
            _CHECK_STRIDES,
            _ENCODE_STRIDES_IN_UNIT,
            _COMPUTE_STRIDES,
            _gemm_compute_block(ctype, prec, ab_constants),
        )
    )
    return code % dict(_x=_x, _y=_y, _a=_a, _zout=_zout, **sub)


# ##### ####### #######
# GEMV
# ##### ####### #######

# GEMV output setup, selected by the static ``inplace`` prop: out-of-place
# copies y into a fresh z; in-place aliases z onto y.
_GEMV_SETUP_Z_OUTPLACE = """
        if ((NULL == %(z)s)
            || (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(y)s)[0]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_SimpleNew(1,
                PyArray_DIMS(%(y)s), PyArray_TYPE(%(y)s));
            if(!%(z)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemv output");
                %(fail)s
            }
        }
        if (beta != 0)
        {
            // If beta is zero, we avoid doing the copy
            if (PyArray_CopyInto(%(z)s, %(y)s) != 0) {
                %(fail)s
            }
        }
"""

_GEMV_SETUP_Z_INPLACE = """
        if (%(z)s != %(y)s)
        {
            Py_XDECREF(%(z)s);
            %(z)s = %(y)s;
            Py_INCREF(%(z)s);
        }
"""

# Single-precision GEMV body. ``%(ctype)s``/``%(prec)s``/``%(elemsize)s`` are
# fixed from the node's (statically known) dtype. ``alpha``/``beta`` keep their
# own stored dtype (Gemv.make_node does not constrain them to match A/x/y).
# ``__SETUP_Z__`` is spliced in (not %-substituted) before the single % pass.
_GEMV_CODE = """

    {
        int elemsize = %(elemsize)s;
        %(ctype)s beta = ((dtype_%(beta)s*)PyArray_DATA(%(beta)s))[0];

        if (PyArray_DIMS(%(A)s)[0] != PyArray_DIMS(%(y)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "Shape mismatch: A.shape[0] != y.shape[0]");
            %(fail)s;
        }
        if (PyArray_DIMS(%(A)s)[1] != PyArray_DIMS(%(x)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "Shape mismatch: A.shape[1] != x.shape[0]");
            %(fail)s;
        }

        // set up z (copy y into a fresh output, or alias y in place)
__SETUP_Z__

        {
            int NA0 = PyArray_DIMS(%(A)s)[0];
            int NA1 = PyArray_DIMS(%(A)s)[1];

            if (NA0 * NA1)
            {
                // Non-empty A matrix

                if (%(must_initialize_y)d && beta == 0)
                {
                    // Most BLAS implementations of GEMV ignore y=nan when beta=0
                    // PyTensor considers that the correct behavior,
                    // and even exploits it to avoid copying or initializing outputs.
                    // By deciding to exploit this, however, it becomes our responsibility
                    // to ensure the behavior even in the rare cases BLAS deviates,
                    // or users will get errors, even for graphs that had no nan to begin with.
                    PyArray_FILLWBYTE(%(z)s, 0);
                }

                /* In the case where A is actually a row or column matrix,
                 * the strides corresponding to the dummy dimension don't matter,
                 * but BLAS requires these to be no smaller than the number of elements in the array.
                 */
                int SA0 = (NA0 > 1) ? (PyArray_STRIDES(%(A)s)[0] / elemsize) : NA1;
                int SA1 = (NA1 > 1) ? (PyArray_STRIDES(%(A)s)[1] / elemsize) : NA0;
                int Sz = PyArray_STRIDES(%(z)s)[0] / elemsize;
                int Sx = PyArray_STRIDES(%(x)s)[0] / elemsize;

                %(ctype)s* A_data = (%(ctype)s*) PyArray_DATA(%(A)s);
                %(ctype)s* x_data = (%(ctype)s*) PyArray_DATA(%(x)s);
                %(ctype)s* z_data = (%(ctype)s*) PyArray_DATA(%(z)s);

                // gemv expects pointers to the beginning of memory arrays,
                // but numpy provides a pointer to the first element,
                // so when the stride is negative, we need to get the last one.
                if (Sx < 0)
                    x_data += (NA1 - 1) * Sx;
                if (Sz < 0)
                    z_data += (NA0 - 1) * Sz;

                if ( ((SA0 < 0) || (SA1 < 0)) && (abs(SA0) == 1 || (abs(SA1) == 1)) )
                {
                    // We can treat the array A as C-or F-contiguous by changing the order of iteration
                    if (SA0 < 0){
                        A_data += (NA0 -1) * SA0;  // Jump to first row
                        SA0 = -SA0;  // Iterate over rows in reverse
                        Sz = -Sz;  // Iterate over y in reverse
                    }
                    if (SA1 < 0){
                        A_data += (NA1 -1) * SA1;  // Jump to first column
                        SA1 = -SA1;  // Iterate over columns in reverse
                        Sx = -Sx;  // Iterate over x in reverse
                    }
                } else if (pytensor_gemv_needs_copy(SA0, SA1))
                {
                    // Array isn't contiguous, we have to make a copy
                    npy_intp dims[2];
                    dims[0] = NA0;
                    dims[1] = NA1;
                    PyArrayObject * A_copy = (PyArrayObject *) PyArray_Copy(%(A)s);
                    if (!A_copy)
                        %(fail)s
                    Py_XDECREF(%(A)s);
                    %(A)s = A_copy;
                    SA0 = (NA0 > 1) ? (PyArray_STRIDES(%(A)s)[0] / elemsize) : NA1;
                    SA1 = (NA1 > 1) ? (PyArray_STRIDES(%(A)s)[1] / elemsize) : NA0;
                    A_data = (%(ctype)s*) PyArray_DATA(%(A)s);
                }

                if (NA0 == 1)
                {
                    // Vector-vector dot product, it seems faster to avoid GEMV
                    %(ctype)s alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    pytensor_%(prec)sgemv_dot_case(NA1, SA1,
                        A_data, x_data, z_data,
                        alpha, beta, Sx);
                }
                else if (SA0 == 1 || SA1 == 1)
                {
                    // C-contiguous or F-contiguous, use GEMV dispatch helper
                    %(ctype)s alpha = ((dtype_%(alpha)s*)PyArray_DATA(%(alpha)s))[0];
                    if (pytensor_%(prec)sgemv_dispatch(NA0, NA1, SA0, SA1,
                            A_data, x_data, z_data,
                            alpha, beta, Sx, Sz) != 0) {
                        %(fail)s
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "A is neither C nor F-contiguous, it should have been copied into a memory-contiguous array;");
                    %(fail)s
                }
            } else
            {
                // Empty A matrix, just scale y by beta
                if (beta != 1.0)
                {
                    npy_intp Sz = PyArray_STRIDES(%(z)s)[0] / elemsize;
                    %(ctype)s* z_data = (%(ctype)s*) PyArray_DATA(%(z)s);
                    for (npy_intp i = 0; i < NA0; ++i)
                    {
                        z_data[i * Sz] = (beta == 0.0) ? 0 : z_data[i * Sz] * beta;
                    }
                }
            }
        }
    }
    """


def gemv_c_code(node, name, inputs, outputs, sub):
    """C code for ``CGemv``: ``z <- beta * y + alpha * dot(A, x)``.

    ``z = y`` if inplace else ``z = y.copy()``; A is a matrix, x and y vectors.
    """
    # Imported lazily to avoid an import cycle (blas_c imports this module).
    from pytensor.tensor.blas.blas_c import must_initialize_y_gemv

    y, alpha, A, x, beta = inputs
    (z,) = outputs
    ctype, prec, elemsize = _check_blas_dtype(node.inputs[0].type.dtype, "CGemv")
    # inplace is a static op prop, so emit only the relevant output setup.
    setup_z = _GEMV_SETUP_Z_INPLACE if node.op.inplace else _GEMV_SETUP_Z_OUTPLACE
    code = _GEMV_CODE.replace("__SETUP_Z__", setup_z)
    return code % dict(
        y=y,
        A=A,
        x=x,
        z=z,
        alpha=alpha,
        beta=beta,
        ctype=ctype,
        prec=prec,
        elemsize=elemsize,
        must_initialize_y=must_initialize_y_gemv(),
        **sub,
    )


# ##### ####### #######
# GER
# ##### ####### #######

# GER rank-1 update Z = A + alpha * outer(x, y). The orchestration lives here
# (dtype-pinned, destructive-pinned, validation elided via Ger.make_node
# guarantees); only the leaf loops / BLAS dispatch stay in ger_helper.h. Rank
# and dtype checks are statically guaranteed and dropped; the shape checks stay
# because shapes are dynamic.

_GER_DIMS_AND_SHAPE_CHECK = """
    {
        npy_intp dims[2];
        dims[0] = PyArray_DIMS(%(A)s)[0];
        dims[1] = PyArray_DIMS(%(A)s)[1];

        if (dims[0] != PyArray_DIMS(%(x)s)[0]) {
            PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[0] != x.shape[0]");
            %(fail)s
        }
        if (dims[1] != PyArray_DIMS(%(y)s)[0]) {
            PyErr_SetString(PyExc_ValueError, "Shape mismatch: A.shape[1] != y.shape[0]");
            %(fail)s
        }
"""

# Copy A into a fresh/contiguous Z, then add the rank-1 update. Used for the
# non-destructive case and for the destructive-but-badly-strided case.
_GER_COPY_BLOCK = """
        int need_alloc = (%(Z)s == NULL)
            || (PyArray_DIMS(%(Z)s)[0] != dims[0])
            || (PyArray_DIMS(%(Z)s)[1] != dims[1])
            || pytensor_ger_needs_copy(
                   PyArray_STRIDES(%(Z)s)[0], PyArray_STRIDES(%(Z)s)[1], %(elemsize)s);
        if (need_alloc) {
            Py_XDECREF(%(Z)s);
            %(Z)s = (PyArrayObject *)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(A)s));
            if (!%(Z)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc ger output");
                %(fail)s
            }
        }
        if (%(Z)s == %(A)s) {
            PyErr_SetString(PyExc_AssertionError, "Z should not be A in copy path");
            %(fail)s
        }
        {
            const %(ctype)s *zdata = (const %(ctype)s *)PyArray_DATA(%(A)s);
            %(ctype)s *zoutdata = (%(ctype)s *)PyArray_DATA(%(Z)s);
            const %(ctype)s *xdata = (const %(ctype)s *)PyArray_DATA(%(x)s);
            const %(ctype)s *ydata = (const %(ctype)s *)PyArray_DATA(%(y)s);
            %(ctype)s alpha = ((%(ctype)s *)PyArray_DATA(%(a)s))[0];
            int Ai = PyArray_STRIDES(%(A)s)[0] / sizeof(%(ctype)s);
            int Aj = PyArray_STRIDES(%(A)s)[1] / sizeof(%(ctype)s);
            int Zi = PyArray_STRIDES(%(Z)s)[0] / sizeof(%(ctype)s);
            int Zj = PyArray_STRIDES(%(Z)s)[1] / sizeof(%(ctype)s);
            int xi = PyArray_STRIDES(%(x)s)[0] / sizeof(%(ctype)s);
            int yj = PyArray_STRIDES(%(y)s)[0] / sizeof(%(ctype)s);
            pytensor_%(prec)sger_manual_copy(dims[0], dims[1],
                zdata, Ai, Aj, zoutdata, Zi, Zj, xdata, xi, ydata, yj, alpha);
        }
"""

# Destructive with good strides: operate in place on A (aliased as Z). Small
# matrices use a manual loop; large ones go through BLAS.
_GER_INPLACE_BLOCK = """
        if (%(Z)s != %(A)s) {
            Py_XDECREF(%(Z)s);
            %(Z)s = %(A)s;
            Py_INCREF(%(Z)s);
        }
        if ((dims[0] * dims[1]) < PYTENSOR_GER_BLAS_THRESHOLD) {
            %(ctype)s *zoutdata = (%(ctype)s *)PyArray_DATA(%(Z)s);
            const %(ctype)s *xdata = (const %(ctype)s *)PyArray_DATA(%(x)s);
            const %(ctype)s *ydata = (const %(ctype)s *)PyArray_DATA(%(y)s);
            %(ctype)s alpha = ((%(ctype)s *)PyArray_DATA(%(a)s))[0];
            int Zi = PyArray_STRIDES(%(Z)s)[0] / sizeof(%(ctype)s);
            int Zj = PyArray_STRIDES(%(Z)s)[1] / sizeof(%(ctype)s);
            int xi = PyArray_STRIDES(%(x)s)[0] / sizeof(%(ctype)s);
            int yj = PyArray_STRIDES(%(y)s)[0] / sizeof(%(ctype)s);
            pytensor_%(prec)sger_manual_inplace(dims[0], dims[1],
                zoutdata, Zi, Zj, xdata, xi, ydata, yj, alpha);
        } else {
            int Nz0 = dims[0];
            int Nz1 = dims[1];
            int Sx = PyArray_STRIDES(%(x)s)[0] / %(elemsize)s;
            int Sy = PyArray_STRIDES(%(y)s)[0] / %(elemsize)s;
            void *x_data = PyArray_DATA(%(x)s);
            void *y_data = PyArray_DATA(%(y)s);
            if (Sx < 0) { x_data = (char *)x_data + (Nz0 - 1) * Sx * %(elemsize)s; }
            if (Sy < 0) { y_data = (char *)y_data + (Nz1 - 1) * Sy * %(elemsize)s; }
            %(ctype)s alpha = ((%(ctype)s *)PyArray_DATA(%(a)s))[0];
            if (pytensor_%(prec)sger_dispatch(Nz0, Nz1,
                    PyArray_STRIDES(%(Z)s)[0], PyArray_STRIDES(%(Z)s)[1], %(elemsize)s,
                    (%(ctype)s *)PyArray_DATA(%(Z)s), (%(ctype)s *)x_data, (%(ctype)s *)y_data,
                    alpha, Sx, Sy) != 0) {
                %(fail)s
            }
        }
"""

_GER_DESTRUCTIVE_DISPATCH = """
        if (pytensor_ger_needs_copy(
                PyArray_STRIDES(%(A)s)[0], PyArray_STRIDES(%(A)s)[1], %(elemsize)s)) {
"""


def ger_c_code(node, name, inputs, outputs, sub):
    """C code for ``CGer``: rank-1 update ``Z = A + alpha * outer(x, y)``."""
    A, a, x, y = inputs
    (Z,) = outputs
    ctype, prec, elemsize = _check_blas_dtype(node.inputs[0].type.dtype, "CGer")
    if node.op.destructive:
        # Branch at runtime on A's strides: copy if badly strided, else in place.
        body = (
            _GER_DIMS_AND_SHAPE_CHECK
            + _GER_DESTRUCTIVE_DISPATCH
            + _GER_COPY_BLOCK
            + "\n        } else {\n"
            + _GER_INPLACE_BLOCK
            + "\n        }\n    }\n"
        )
    else:
        # Non-destructive: always copy A into Z.
        body = _GER_DIMS_AND_SHAPE_CHECK + _GER_COPY_BLOCK + "\n    }\n"
    return body % dict(
        A=A, a=a, x=x, y=y, Z=Z, ctype=ctype, prec=prec, elemsize=elemsize, **sub
    )
