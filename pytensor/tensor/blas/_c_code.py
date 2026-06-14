# C source for the BLAS ops, kept as Python strings so the package ships no
# on-disk .c/.h data files (BLAS codegen is 100% generated from Python). These
# are raw strings: backslashes are literal C, not Python escapes.
#
# ALT_BLAS_TEMPLATE is %-substituted (float/double); every other constant is
# emitted verbatim into c_support_code, so a literal "%" in them is intentional.

FORTRAN_BLAS = r"""/*
 * Fortran BLAS interface declarations for PyTensor.
 *
 * These are the extern "C" declarations for the Fortran BLAS routines
 * (with trailing underscore convention). Used by GemmRelated, CGemv, CGer, etc.
 */

#ifndef PYTENSOR_FORTRAN_BLAS_H
#define PYTENSOR_FORTRAN_BLAS_H

extern "C"
{

    void xerbla_(char*, void *);

/***********/
/* Level 1 */
/***********/

/* Single Precision */

    void srot_(const int*, float *, const int*, float *, const int*, const float *, const float *);
    void srotg_(float *,float *,float *,float *);
    void srotm_( const int*, float *, const int*, float *, const int*, const float *);
    void srotmg_(float *,float *,float *,const float *, float *);
    void sswap_( const int*, float *, const int*, float *, const int*);
    void scopy_( const int*, const float *, const int*, float *, const int*);
    void saxpy_( const int*, const float *, const float *, const int*, float *, const int*);
    float sdot_(const int*, const float *, const int*, const float *, const int*);
    void sdot_sub_(const int*, const float *, const int*, const float *, const int*, float *);
    void sdsdot_sub_( const int*, const float *, const float *, const int*, const float *, const int*, float *);
    void sscal_( const int*, const float *, float *, const int*);
    void snrm2_sub_( const int*, const float *, const int*, float *);
    void sasum_sub_( const int*, const float *, const int*, float *);
    void isamax_sub_( const int*, const float * , const int*, const int*);

/* Double Precision */

    void drot_(const int*, double *, const int*, double *, const int*, const double *, const double *);
    void drotg_(double *,double *,double *,double *);
    void drotm_( const int*, double *, const int*, double *, const int*, const double *);
    void drotmg_(double *,double *,double *,const double *, double *);
    void dswap_( const int*, double *, const int*, double *, const int*);
    void dcopy_( const int*, const double *, const int*, double *, const int*);
    void daxpy_( const int*, const double *, const double *, const int*, double *, const int*);
    void dswap_( const int*, double *, const int*, double *, const int*);
    double ddot_(const int*, const double *, const int*, const double *, const int*);
    void dsdot_sub_(const int*, const float *, const int*, const float *, const int*, double *);
    void ddot_sub_( const int*, const double *, const int*, const double *, const int*, double *);
    void dscal_( const int*, const double *, double *, const int*);
    void dnrm2_sub_( const int*, const double *, const int*, double *);
    void dasum_sub_( const int*, const double *, const int*, double *);
    void idamax_sub_( const int*, const double * , const int*, const int*);

/* Single Complex Precision */

    void cswap_( const int*, void *, const int*, void *, const int*);
    void ccopy_( const int*, const void *, const int*, void *, const int*);
    void caxpy_( const int*, const void *, const void *, const int*, void *, const int*);
    void cswap_( const int*, void *, const int*, void *, const int*);
    void cdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void cdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void cscal_( const int*, const void *, void *, const int*);
    void icamax_sub_( const int*, const void *, const int*, const int*);
    void csscal_( const int*, const float *, void *, const int*);
    void scnrm2_sub_( const int*, const void *, const int*, float *);
    void scasum_sub_( const int*, const void *, const int*, float *);

/* Double Complex Precision */

    void zswap_( const int*, void *, const int*, void *, const int*);
    void zcopy_( const int*, const void *, const int*, void *, const int*);
    void zaxpy_( const int*, const void *, const void *, const int*, void *, const int*);
    void zswap_( const int*, void *, const int*, void *, const int*);
    void zdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void zdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void zdscal_( const int*, const double *, void *, const int*);
    void zscal_( const int*, const void *, void *, const int*);
    void dznrm2_sub_( const int*, const void *, const int*, double *);
    void dzasum_sub_( const int*, const void *, const int*, double *);
    void izamax_sub_( const int*, const void *, const int*, const int*);

/***********/
/* Level 2 */
/***********/

/* Single Precision */

    void sgemv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void sgbmv_(char*, const int*, const int*, const int*, const int*, const float *,  const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void ssymv_(char*, const int*, const float *, const float *, const int*, const float *,  const int*, const float *, float *, const int*);
    void ssbmv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void sspmv_(char*, const int*, const float *, const float *, const float *, const int*, const float *, float *, const int*);
    void strmv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
    void stbmv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
    void strsv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
    void stbsv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
    void stpmv_( char*, char*, char*, const int*, const float *, float *, const int*);
    void stpsv_( char*, char*, char*, const int*, const float *, float *, const int*);
    void sger_( const int*, const int*, const float *, const float *, const int*, const float *, const int*, float *, const int*);
    void ssyr_(char*, const int*, const float *, const float *, const int*, float *, const int*);
    void sspr_(char*, const int*, const float *, const float *, const int*, float *);
    void sspr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *);
    void ssyr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *, const int*);

/* Double Precision */

    void dgemv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dgbmv_(char*, const int*, const int*, const int*, const int*, const double *,  const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dsymv_(char*, const int*, const double *, const double *, const int*, const double *,  const int*, const double *, double *, const int*);
    void dsbmv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dspmv_(char*, const int*, const double *, const double *, const double *, const int*, const double *, double *, const int*);
    void dtrmv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
    void dtbmv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
    void dtrsv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
    void dtbsv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
    void dtpmv_( char*, char*, char*, const int*, const double *, double *, const int*);
    void dtpsv_( char*, char*, char*, const int*, const double *, double *, const int*);
    void dger_( const int*, const int*, const double *, const double *, const int*, const double *, const int*, double *, const int*);
    void dsyr_(char*, const int*, const double *, const double *, const int*, double *, const int*);
    void dspr_(char*, const int*, const double *, const double *, const int*, double *);
    void dspr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *);
    void dsyr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *, const int*);

/* Single Complex Precision */

    void cgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void cgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void chemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void chbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void chpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
    void ctrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ctbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ctpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
    void ctrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ctbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ctpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
    void cgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void cgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
    void cher_(char*, const int*, const float *, const void *, const int*, void *, const int*);
    void cher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void chpr_(char*, const int*, const float *, const void *, const int*, void *);
    void chpr2_(char*, const int*, const float *, const void *, const int*, const void *, const int*, void *);

/* Double Complex Precision */

    void zgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zhemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zhbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zhpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
    void ztrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ztbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ztpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
    void ztrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ztbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ztpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
    void zgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void zgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
    void zher_(char*, const int*, const double *, const void *, const int*, void *, const int*);
    void zher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void zhpr_(char*, const int*, const double *, const void *, const int*, void *);
    void zhpr2_(char*, const int*, const double *, const void *, const int*, const void *, const int*, void *);

/***********/
/* Level 3 */
/***********/

/* Single Precision */

    void sgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void ssymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void ssyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
    void ssyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void strmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
    void strsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

/* Double Precision */

    void dgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
    void dsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dtrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
    void dtrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

/* Single Complex Precision */

    void cgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void csymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void chemm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void csyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
    void cherk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
    void csyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void cher2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void ctrmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
    void ctrsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

/* Double Complex Precision */

    void zgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zhemm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
    void zherk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
    void zsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zher2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void ztrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
    void ztrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

}

#endif /* PYTENSOR_FORTRAN_BLAS_H */

"""

ALT_BLAS_COMMON = r"""/** C Implementation (with NumPy back-end) of BLAS functions used in PyTensor.
 * Used instead of BLAS when PyTensor flag ``blas__ldflags`` is empty.
 * This file contains some useful header code not templated.
 * File alt_blas_template.c currently contains template code for:
 * - [sd]gemm_
 * - [sd]gemv_
 * - [sd]dot_
 **/

#define alt_fatal_error(message) { if (PyErr_Occurred()) PyErr_Print(); if(message != NULL) fprintf(stderr, message); exit(-1); }

#define alt_trans_to_bool(trans)  (*trans != 'N' && *trans != 'n')

/**Template code for BLAS functions follows in file alt_blas_template.c
 * (as Python string to be used with old formatting).
 * PARAMETERS:
 * float_type: "float" or "double".
 * float_size: 4 for float32 (sgemm_), 8 for float64 (dgemm_).
 * npy_float: "NPY_FLOAT32" or "NPY_FLOAT64".
 * precision: "s" for single, "d" for double.
 * See blas_headers.py for current use.**/
"""

ALT_BLAS_TEMPLATE = r"""/** Alternative template NumPy-based implementation of BLAS functions used in PyTensor. **/

/* Compute matrix[i][j] = scalar for every position (i, j) in matrix. */
void alt_numpy_memset_inplace_%(float_type)s(PyArrayObject* matrix, const %(float_type)s* scalar) {
    if (PyArray_IS_C_CONTIGUOUS(matrix) && *scalar == (char)(*scalar)) {
        // This will use memset.
        PyArray_FILLWBYTE(matrix, (char)(*scalar));
        return;
    }
    NpyIter* iterator = NpyIter_New(matrix,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix for a memory assignation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((%(float_type)s*)data) = *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}

/* Scalar * Matrix function.
 * Computes: matrix = scalar * matrix. */
void alt_numpy_scale_matrix_inplace_%(float_type)s(const %(float_type)s* scalar, PyArrayObject* matrix) {
    if (*scalar == 1)
        return;
    if (*scalar == 0) {
        alt_numpy_memset_inplace_%(float_type)s(matrix, scalar);
        return;
    }
    NpyIter* iterator = NpyIter_New(matrix,
        NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if(iterator == NULL)
        alt_fatal_error("Unable to iterate over a matrix "
                        "for a scalar * matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
    char** data_ptr = NpyIter_GetDataPtrArray(iterator);
    npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
    npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
    do {
        char* data = *data_ptr;
        npy_intp stride = *stride_ptr;
        npy_intp count = *innersize_ptr;
        while(count) {
            *((%(float_type)s*)data) *= *scalar;
            data += stride;
            --count;
        }
    } while(get_next(iterator));
    NpyIter_Deallocate(iterator);
}

/* Matrix + Matrix function.
 * Computes: matrix2 = (scalar1 * matrix1) + (scalar2 * matrix2) */
void alt_numpy_matrix_extended_sum_inplace_%(float_type)s(
        const %(float_type)s* scalar1, PyArrayObject* matrix1,
        const %(float_type)s* scalar2, PyArrayObject* matrix2
) {
    if (*scalar1 == 0 && *scalar2 == 0) {
        alt_numpy_memset_inplace_%(float_type)s(matrix2, scalar2);
        return;
    }
    if (*scalar1 == 0) {
        alt_numpy_scale_matrix_inplace_%(float_type)s(scalar2, matrix2);
        return;
    }
    PyArrayObject* op[2]       = {matrix1, matrix2};
    npy_uint32     op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_READWRITE};
    npy_uint32     flags       = 0;
    NpyIter*       iterators   = NpyIter_MultiNew(
            2, op, flags, NPY_CORDER, NPY_NO_CASTING, op_flags, NULL);
    if(iterators == NULL)
        alt_fatal_error("Unable to iterate over some matrices "
                        "for matrix + matrix operation.");
    NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterators, NULL);
    char** data_ptr_array = NpyIter_GetDataPtrArray(iterators);
    if (*scalar2 == 0) {
        do {
            %(float_type)s* from_matrix1 = (%(float_type)s*)data_ptr_array[0];
            %(float_type)s* from_matrix2 = (%(float_type)s*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1);
        } while(get_next(iterators));
    } else {
        do {
            %(float_type)s* from_matrix1 = (%(float_type)s*)data_ptr_array[0];
            %(float_type)s* from_matrix2 = (%(float_type)s*)data_ptr_array[1];
            *from_matrix2 = (*scalar1)*(*from_matrix1) + (*scalar2)*(*from_matrix2);
        } while(get_next(iterators));
    }
    NpyIter_Deallocate(iterators);
}

/* NumPy Wrapping function. Wraps a data into a NumPy's PyArrayObject.
 * By default, data is considered as Fortran-style array (column by column).
 * If to_transpose, data will be considered as C-style array (row by row)
 * with dimensions reversed. */
PyObject* alt_op_%(float_type)s(int to_transpose, %(float_type)s* M, int nrow, int ncol, int LDM, int numpyFlags) {
    npy_intp dims[2];
    npy_intp strides[2];
    if(to_transpose) {
        dims[0] = ncol;
        dims[1] = nrow;
        strides[0] = LDM * %(float_size)d;
        strides[1] = %(float_size)d;
    } else {
        dims[0] = nrow;
        dims[1] = ncol;
        strides[0] = %(float_size)d;
        strides[1] = LDM * %(float_size)d;
    }
    return PyArray_New(&PyArray_Type, 2, dims, %(npy_float)s, strides, M, 0, numpyFlags, NULL);
}

/* Special wrapping case used for matrix C in gemm_ implementation. */
inline PyObject* alt_wrap_fortran_writeable_matrix_%(float_type)s(
    %(float_type)s* matrix, const int* nrow, const int* ncol, const int* LD
) {
    npy_intp dims[2] = {*nrow, *ncol};
    npy_intp strides[2] = {%(float_size)d, (*LD) * %(float_size)d};
    return PyArray_New(&PyArray_Type, 2, dims, %(npy_float)s, strides, matrix, 0, NPY_ARRAY_WRITEABLE, NULL);
}

/* gemm */
void %(precision)sgemm_(
    char* TRANSA, char* TRANSB, const int* M, const int* N, const int* K,
    const %(float_type)s* ALPHA, %(float_type)s* A, const int* LDA,
    %(float_type)s* B, const int* LDB, const %(float_type)s* BETA,
    %(float_type)s* C, const int* LDC
) {
    if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
        alt_fatal_error("The integer arguments passed to %(precision)sgemm_ must all be at least 0.");
    /* If M or N is null, there is nothing to do with C,
     * as C should contain M*N == 0 items. */
    if(*M == 0 || *N == 0)
        return;
    int nrowa, ncola, nrowb, ncolb;
    int to_transpose_A = alt_trans_to_bool(TRANSA);
    int to_transpose_B = alt_trans_to_bool(TRANSB);
    if(to_transpose_A) {
        nrowa = *K;
        ncola = *M;
    } else {
        nrowa = *M;
        ncola = *K;
    }
    if(to_transpose_B) {
        nrowb = *N;
        ncolb = *K;
    } else {
        nrowb = *K;
        ncolb = *N;
    }
    int computation_flags;
    void* computation_pointer;
    npy_intp* computation_strides;
    npy_intp computation_dims[2] = {*N, *M};
    npy_intp default_computation_strides[2] = {(*LDC) * %(float_size)d, %(float_size)d};
    if(*BETA == 0 && *LDC == *M) {
        /* BETA == 0, so C is never read.
         * LDC == M, so C is contiguous in memory
         * (that condition is needed for dot operation, se below).
         * Then we can compute ALPHA*op(A)*op(B) directly in C. */
        computation_flags = NPY_ARRAY_WRITEABLE;
        computation_pointer = C;
        computation_strides = default_computation_strides;
    } else {
        /* Either BETA != 0 (C will be read)
         * or LDC != M (C is not read but is not contiguous in memory).
         * Then in both cases, we need to allocate a new memory
         * to compute ALPHA*op(A)*op(B). */
        computation_flags = 0;
        computation_pointer = NULL;
        computation_strides = NULL;
    }
    /* The memory buffer used to compute op(A)*op(B) (either C or
     * new allocated buffer) will be considered as C-contiguous because
     * the 3rd parameter of PyArray_MatrixProduct2 (used below)
     * expects a C-contiguous array.
     * Also, to avoid some memory copy, transposition conditions
     * for A and B will be reversed, so that the buffer will contain
     * C-contiguous opB_transposed * opA_transposed (N*M matrix).
     * After that, the code that uses the buffer (either the code calling
     * this function, or this function if BETA != 0) just has to
     * consider the buffer as a F-contiguous M*N matrix, so that
     * it will get the transposed of op_B_transposed * op_A_transposed,
     * that is op_A * op_B (M*N matrix) as expected. */
    PyObject* opA_transposed = alt_op_%(float_type)s(!to_transpose_A, A, nrowa, ncola, *LDA, 0);
    PyObject* opB_transposed = alt_op_%(float_type)s(!to_transpose_B, B, nrowb, ncolb, *LDB, 0);
    PyObject* opB_trans_dot_opA_trans = PyArray_New(&PyArray_Type, 2, computation_dims, %(npy_float)s,
                                                    computation_strides, computation_pointer, 0,
                                                    computation_flags, NULL);
    PyArray_MatrixProduct2(opB_transposed, opA_transposed, (PyArrayObject*)opB_trans_dot_opA_trans);
    /* PyArray_MatrixProduct2 adds a reference to the output array,
     * which we need to remove to avoid a memory leak. */
    Py_XDECREF(opB_trans_dot_opA_trans);
    if(*BETA == 0) {
        if(*ALPHA != 1.0)
            alt_numpy_scale_matrix_inplace_%(float_type)s(ALPHA, (PyArrayObject*)opB_trans_dot_opA_trans);
        if(*LDC != *M) {
            /* A buffer has been created to compute ALPHA*op(A)*op(B),
             * so we must copy it to the real output, that is C. */
            PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_%(float_type)s(C, M, N, LDC);
            PyObject* alpha_opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrix_C, (PyArrayObject*)alpha_opA_dot_opB))
                alt_fatal_error("NumPy %(precision)sgemm_ implementation: unable to copy ALPHA*op(A)*op(B) into C when BETA == 0.");
            Py_XDECREF(alpha_opA_dot_opB);
            Py_XDECREF(matrix_C);
        }
    } else {
        /* C is read, so we must consider it as Fortran-style matrix. */
        PyObject* matrix_C = alt_wrap_fortran_writeable_matrix_%(float_type)s(C, M, N, LDC);
        PyObject* opA_dot_opB = PyArray_Transpose((PyArrayObject*)opB_trans_dot_opA_trans, NULL);
        alt_numpy_matrix_extended_sum_inplace_%(float_type)s(ALPHA, (PyArrayObject*)opA_dot_opB,
                                                             BETA, (PyArrayObject*)matrix_C);
        Py_XDECREF(opA_dot_opB);
        Py_XDECREF(matrix_C);
    }
    Py_XDECREF(opB_trans_dot_opA_trans);
    Py_XDECREF(opB_transposed);
    Py_XDECREF(opA_transposed);
}

/* gemv */
void %(precision)sgemv_(
    char* TRANS,
    const int* M,
    const int* N,
    const %(float_type)s* ALPHA,
    %(float_type)s* A,
    const int* LDA,
    %(float_type)s* x,
    const int* incx,
    const %(float_type)s* BETA,
    %(float_type)s* y,
    const int* incy
) {
    /**
    If TRANS is 'n' or 'N', computes:
        y = ALPHA * A * x + BETA * y
    Else, computes:
        y = ALPHA * A.T * x + BETA * y
    A is a M*N matrix, A.T is A transposed
    x, y are vectors
    ALPHA, BETA are scalars
    **/

    // If alpha == 0 and beta == 1, we have nothing to do, as alpha*A*x + beta*y == y.
    if (*ALPHA == 0 && *BETA == 1)
        return;
    if (*M < 0 || *N < 0 || *LDA < 0)
        alt_fatal_error("NumPy %(precision)sgemv_ implementation: M, N and LDA must be at least 0.");
    if (*incx == 0 || *incy == 0)
        alt_fatal_error("NumPy %(precision)sgemv_ implementation: incx and incy must not be 0.");
    int transpose = alt_trans_to_bool(TRANS);
    int size_x = 0, size_y = 0;
    if (transpose) {
        size_x = *M;
        size_y = *N;
    } else {
        size_x = *N;
        size_y = *M;
    }
    if (*M == 0 || *N == 0) {
        /* A contains M * N == 0 values. y should be empty too, and we have nothing to do. */
        if (size_y != 0)
            alt_fatal_error("NumPy %(precision)sgemv_ implementation: the output vector should be empty.");
        return;
    }
    /* Vector pointers points to the beginning of memory (see function `pytensor.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*incx < 0)
        x += (size_x - 1) * (-*incx);
    if (*incy < 0)
        y += (size_y - 1) * (-*incy);
    PyObject* matrixA = alt_op_%(float_type)s(transpose, A, *M, *N, *LDA, 0);
    PyObject* matrixX = alt_op_%(float_type)s(1, x, 1, size_x, *incx, 0);
    PyObject* matrixY = alt_op_%(float_type)s(1, y, 1, size_y, *incy, NPY_ARRAY_WRITEABLE);
    if (matrixA == NULL || matrixX == NULL || matrixY == NULL)
        alt_fatal_error("NumPy %(precision)sgemv_ implementation: unable to wrap A, x or y arrays.")
    if (*ALPHA == 0) {
        // Just BETA * y
        alt_numpy_scale_matrix_inplace_%(float_type)s(BETA, (PyArrayObject*)matrixY);
    } else if (*BETA == 0) {
        // We can directly compute alpha * A * x into y if y is C-contiguous.
        if (PyArray_IS_C_CONTIGUOUS((PyArrayObject*)matrixY)) {
            PyArray_MatrixProduct2(matrixA, matrixX, (PyArrayObject*)matrixY);
            // PyArray_MatrixProduct2 adds an extra reference to the output array.
            Py_XDECREF(matrixY);
            alt_numpy_scale_matrix_inplace_%(float_type)s(ALPHA, (PyArrayObject*)matrixY);
        } else {
            // If y is not contiguous, we need a temporar workspace.
            PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
            if (tempAX == NULL)
                alt_fatal_error("NumPy %(precision)sgemv_ implementation: Unable to get matrix product.");
            alt_numpy_scale_matrix_inplace_%(float_type)s(ALPHA, (PyArrayObject*)tempAX);
            if(0 != PyArray_CopyInto((PyArrayObject*)matrixY, (PyArrayObject*)tempAX)) {
                alt_fatal_error("NumPy %(precision)sgemv_ implementation: unable to update output.");
            }
            Py_XDECREF(tempAX);
        }
    } else {
        // We must perform full computation.
        PyObject* tempAX = PyArray_MatrixProduct(matrixA, matrixX);
        if (tempAX == NULL)
            alt_fatal_error("NumPy %(precision)sgemv_ implementation: unable to get matrix product.");
        // ALPHA * (A * x) + BETA * y.
        alt_numpy_matrix_extended_sum_inplace_%(float_type)s(ALPHA, (PyArrayObject*)tempAX,
                                                             BETA, (PyArrayObject*)matrixY);
        Py_XDECREF(tempAX);
    }
    Py_XDECREF(matrixY);
    Py_XDECREF(matrixX);
    Py_XDECREF(matrixA);
}

/* dot */
%(float_type)s %(precision)sdot_(
    const int* N,
    %(float_type)s *SX,
    const int *INCX,
    %(float_type)s *SY,
    const int *INCY
) {
    if (*N < 0)
        alt_fatal_error("NumPy %(precision)sdot_ implementation: N must be at least 0.");
    if (*INCX == 0 || *INCY == 0)
        alt_fatal_error("NumPy %(precision)sdot_ implementation: INCX and INCY must not be 0.");
    %(float_type)s result = 0;
    int one = 1;
    /* Vector pointers points to the beginning of memory (see function `pytensor.tensor.blas_c.gemv_c_code`).
     * NumPy seems to expect that pointers points to the first element of the array. */
    if (*INCX < 0)
        SX += (*N - 1) * (-*INCX);
    if (*INCY < 0)
        SY += (*N - 1) * (-*INCY);
    // Create vector_x with shape (1, N)
    PyObject* vector_x = alt_op_%(float_type)s(0, SX, 1, *N, *INCX, 0);
    // Create vector_y with shape (N, 1)
    PyObject* vector_y = alt_op_%(float_type)s(1, SY, 1, *N, *INCY, 0);
    // Create output scalar z with shape (1, 1) to wrap `result`.
    PyArrayObject* dot_product = (PyArrayObject*)alt_wrap_fortran_writeable_matrix_%(float_type)s(&result, &one, &one, &one);

    if (vector_x == NULL || vector_y == NULL || dot_product == NULL)
        alt_fatal_error("NumPy %(precision)sdot_ implementation: unable to wrap x, y or output arrays.");

    // Compute matrix product: (1, N) * (N, 1) => (1, 1)
    PyArray_MatrixProduct2(vector_x, vector_y, dot_product);
    // PyArray_MatrixProduct2 adds an extra reference to the output array.
    Py_XDECREF(dot_product);

    if (PyErr_Occurred())
        alt_fatal_error("NumPy %(precision)sdot_ implementation: unable to compute dot.");

    Py_XDECREF(dot_product);
    Py_XDECREF(vector_y);
    Py_XDECREF(vector_x);
    return result;
}
"""

MKL_THREADS = r"""/*
 * MKL threads interface declarations for PyTensor.
 */

#ifndef PYTENSOR_MKL_THREADS_H
#define PYTENSOR_MKL_THREADS_H

extern "C"
{
    int     MKL_Set_Num_Threads_Local(int);
    #define mkl_set_num_threads_local   MKL_Set_Num_Threads_Local

    void    MKL_Set_Num_Threads(int);
    #define mkl_set_num_threads         MKL_Set_Num_Threads

    int     MKL_Get_Max_Threads(void);
    #define mkl_get_max_threads         MKL_Get_Max_Threads

    int     MKL_Domain_Set_Num_Threads(int, int);
    #define mkl_domain_set_num_threads  MKL_Domain_Set_Num_Threads

    int     MKL_Domain_Get_Max_Threads(int);
    #define mkl_domain_get_max_threads  MKL_Domain_Get_Max_Threads

    void    MKL_Set_Dynamic(int);
    #define mkl_set_dynamic             MKL_Set_Dynamic

    int     MKL_Get_Dynamic(void);
    #define mkl_get_dynamic             MKL_Get_Dynamic
}

#endif /* PYTENSOR_MKL_THREADS_H */

"""

OPENBLAS_THREADS = r"""/*
 * OpenBLAS threads interface declarations for PyTensor.
 */

#ifndef PYTENSOR_OPENBLAS_THREADS_H
#define PYTENSOR_OPENBLAS_THREADS_H

extern "C"
{
    void openblas_set_num_threads(int);
    void goto_set_num_threads(int);
    int openblas_get_num_threads(void);
}

#endif /* PYTENSOR_OPENBLAS_THREADS_H */

"""

MACOS_SDOT_TEST = r"""/*
 * Test program to detect the macOS BLAS sdot_ bug.
 *
 * Apple's Accelerate framework has a long-standing bug where the Fortran
 * interface sdot_() returns incorrect values. This test computes a simple
 * dot product and checks if the result is correct.
 *
 * Expected result: 0*0 + 1*1 + 2*2 + 3*3 + 4*4 = 30
 * Returns 0 if correct, -1 if bug is present.
 */

extern "C" float sdot_(int*, float*, int*, float*, int*);

int main(int argc, char** argv)
{
    int Nx = 5;
    int Sx = 1;
    float x[5] = {0, 1, 2, 3, 4};
    float r = sdot_(&Nx, x, &Sx, x, &Sx);

    if ((r - 30.f) > 1e-6 || (r - 30.f) < -1e-6)
    {
        return -1;
    }
    return 0;
}

"""

MACOS_SDOT_FIX_TEST = r"""/*
 * Test program to verify the macOS BLAS sdot_ bug workaround.
 *
 * This defines a static sdot_ wrapper that uses cblas_sdot internally,
 * then tests if it returns the correct result. The C interface cblas_sdot()
 * works correctly even when the Fortran sdot_() is buggy.
 *
 * Expected result: 0*0 + 1*1 + 2*2 + 3*3 + 4*4 = 30
 * Returns 0 if workaround works, -1 if it fails.
 */

extern "C" float cblas_sdot(int, float*, int, float*, int);

static float sdot_(int* Nx, float* x, int* Sx, float* y, int* Sy)
{
    return cblas_sdot(*Nx, x, *Sx, y, *Sy);
}

int main(int argc, char** argv)
{
    int Nx = 5;
    int Sx = 1;
    float x[5] = {0, 1, 2, 3, 4};
    float r = sdot_(&Nx, x, &Sx, x, &Sx);

    if ((r - 30.f) > 1e-6 || (r - 30.f) < -1e-6)
    {
        return -1;
    }
    return 0;
}

"""

MACOS_SDOT_WORKAROUND = r"""/*
 * macOS sdot_ bug workaround.
 *
 * Apple's Accelerate framework has a bug where the Fortran sdot_() interface
 * returns incorrect values. This wrapper uses cblas_sdot() instead, which
 * works correctly.
 */

extern "C" float cblas_sdot(int, float*, int, float*, int);
static float sdot_(int* Nx, float* x, int* Sx, float* y, int* Sy)
{
    return cblas_sdot(*Nx, x, *Sx, y, *Sy);
}

"""

MACOS_SDOT_ERROR = r"""/*
 * macOS sdot_ bug fatal error stub.
 *
 * When the sdot_ bug is detected but no workaround is available,
 * this stub ensures we fail loudly rather than silently returning
 * incorrect results.
 */

static float sdot_(int* Nx, float* x, int* Sx, float* y, int* Sy)
{
    fprintf(stderr,
        "FATAL: The implementation of BLAS SDOT "
        "routine in your system has a bug that "
        "makes it return wrong results.\n"
        "You can work around this bug by using a "
        "different BLAS library, or disabling BLAS\n");
    assert(0);
}

"""

GEMM_HELPER = r"""/*
 * GEMM helper functions for PyTensor.
 *
 * This file contains the core GEMM dispatch logic extracted from the
 * Python code generation templates. The goal is to have real C code
 * that IDEs can parse, with minimal dynamic parts.
 */

#ifndef PYTENSOR_GEMM_HELPER_H
#define PYTENSOR_GEMM_HELPER_H

#include <Python.h>
#include <numpy/arrayobject.h>

#ifndef MOD
#define MOD %
#endif

/*
 * Compute strides for a contiguous array.
 * Used when PyArray_STRIDES returns invalid values (e.g., for size-0 arrays).
 */
static inline void compute_strides(npy_intp *shape, int ndim, int type_size, npy_intp *res) {
    res[ndim - 1] = type_size;
    for (int i = ndim - 1; i > 0; i--) {
        npy_intp s = shape[i];
        res[i - 1] = res[i] * (s > 0 ? s : 1);
    }
}

/*
 * Encode the stride structure of three 2D arrays into a single integer.
 *
 * For each array, we encode:
 *   0x0 = row-major (last stride == type_size) or single column
 *   0x1 = column-major (first stride == type_size) or single row
 *   0x2 = neither (will trigger error)
 *
 * The encoding is: (x_code << 8) | (y_code << 4) | (z_code << 0)
 */
static inline int pytensor_encode_gemm_strides(
    npy_intp *Nx, npy_intp *Sx,
    npy_intp *Ny, npy_intp *Sy,
    npy_intp *Nz, npy_intp *Sz,
    int type_size
) {
    int unit = 0;
    unit |= ((Sx[1] == type_size || Nx[1] == 1) ? 0x0 : (Sx[0] == type_size || Nx[0] == 1) ? 0x1 : 0x2) << 8;
    unit |= ((Sy[1] == type_size || Ny[1] == 1) ? 0x0 : (Sy[0] == type_size || Ny[0] == 1) ? 0x1 : 0x2) << 4;
    unit |= ((Sz[1] == type_size || Nz[1] == 1) ? 0x0 : (Sz[0] == type_size || Nz[0] == 1) ? 0x1 : 0x2) << 0;
    return unit;
}

/*
 * Compute BLAS-compatible strides from NumPy strides.
 *
 * BLAS requires leading dimensions to be >= 1 and not smaller than
 * the number of elements in that dimension. For vectors or empty
 * matrices, we need to compute valid dummy strides.
 */
static inline void pytensor_compute_gemm_strides(
    npy_intp *Nx, npy_intp *Sx, int *sx_0, int *sx_1,
    npy_intp *Ny, npy_intp *Sy, int *sy_0, int *sy_1,
    npy_intp *Nz, npy_intp *Sz, int *sz_0, int *sz_1,
    int type_size
) {
    *sx_0 = (Nx[0] > 1) ? Sx[0] / type_size : (Nx[1] + 1);
    *sx_1 = (Nx[1] > 1) ? Sx[1] / type_size : (Nx[0] + 1);
    *sy_0 = (Ny[0] > 1) ? Sy[0] / type_size : (Ny[1] + 1);
    *sy_1 = (Ny[1] > 1) ? Sy[1] / type_size : (Ny[0] + 1);
    *sz_0 = (Nz[0] > 1) ? Sz[0] / type_size : (Nz[1] + 1);
    *sz_1 = (Nz[1] > 1) ? Sz[1] / type_size : (Nz[0] + 1);
}

/*
 * Call sgemm_ with the appropriate transpose flags based on stride encoding.
 *
 * Returns 0 on success, -1 on error (with Python exception set).
 */
static inline int pytensor_sgemm_dispatch(
    int unit,
    float *x, float *y, float *z,
    float a, float b,
    int Nz0, int Nz1, int Nx1,
    int sx_0, int sx_1, int sy_0, int sy_1, int sz_0, int sz_1
) {
    char N = 'N';
    char T = 'T';

    switch (unit) {
        case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
        case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
        case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
        case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
        case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
        case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
        case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
        case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
        default:
            PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride");
            return -1;
    }
    return 0;
}

/*
 * Call dgemm_ with the appropriate transpose flags based on stride encoding.
 *
 * Returns 0 on success, -1 on error (with Python exception set).
 */
static inline int pytensor_dgemm_dispatch(
    int unit,
    double *x, double *y, double *z,
    double a, double b,
    int Nz0, int Nz1, int Nx1,
    int sx_0, int sx_1, int sy_0, int sy_1, int sz_0, int sz_1
) {
    char N = 'N';
    char T = 'T';

    switch (unit) {
        case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
        case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
        case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
        case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
        case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
        case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
        case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
        case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
        default:
            PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride");
            return -1;
    }
    return 0;
}

/*
 * Check if an array needs to be copied to make it BLAS-compatible.
 *
 * BLAS requires arrays to have at least one unit stride and valid
 * (non-negative, properly aligned) strides.
 */
static inline int pytensor_needs_copy_for_blas(npy_intp *N, npy_intp *S, int type_size) {
    return (S[0] < 1) || (S[1] < 1)
        || (S[0] MOD type_size) || (S[1] MOD type_size)
        || ((S[0] != type_size) && (S[1] != type_size));
}

#endif /* PYTENSOR_GEMM_HELPER_H */

"""

GEMV_HELPER = r"""/*
 * GEMV helper functions for PyTensor.
 *
 * This file contains GEMV dispatch logic extracted from Python code generation
 * templates. The goal is to have real C code that IDEs can parse.
 *
 * GEMV computes: z <- beta * y + alpha * dot(A, x)
 * where A is a matrix, x and y are vectors.
 */

#ifndef PYTENSOR_GEMV_HELPER_H
#define PYTENSOR_GEMV_HELPER_H

#include <Python.h>
#include <numpy/arrayobject.h>

/*
 * Check if a matrix needs to be copied to be BLAS-compatible.
 *
 * Returns 1 if copy needed, 0 if matrix can be used directly.
 * A matrix can be used directly if:
 * - It's C-contiguous (SA1 == 1) or F-contiguous (SA0 == 1)
 * - OR strides are negative but can be handled by reversing iteration
 */
static inline int pytensor_gemv_needs_copy(int SA0, int SA1) {
    /* Can handle negative strides by reversing iteration if one stride is ±1 */
    if ((SA0 < 0 || SA1 < 0) && (abs(SA0) == 1 || abs(SA1) == 1)) {
        return 0;
    }
    /* Otherwise need copy if neither stride is 1 or if strides are negative */
    return (SA0 < 0) || (SA1 < 0) || ((SA0 != 1) && (SA1 != 1));
}

/*
 * Call sgemv_ for float matrix-vector multiply.
 *
 * Handles both C-contiguous and F-contiguous layouts.
 * For C-contiguous (SA1 == 1): uses transpose
 * For F-contiguous (SA0 == 1): no transpose
 *
 * Returns 0 on success, -1 on error.
 */
static inline int pytensor_sgemv_dispatch(
    int NA0, int NA1,
    int SA0, int SA1,
    float *A_data, float *x_data, float *z_data,
    float alpha, float beta,
    int Sx, int Sz
) {
    if (SA0 == 1) {
        /* F-contiguous */
        char NOTRANS = 'N';
        sgemv_(&NOTRANS, &NA0, &NA1,
               &alpha, A_data, &SA1,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else if (SA1 == 1) {
        /* C-contiguous */
        char TRANS = 'T';
        sgemv_(&TRANS, &NA1, &NA0,
               &alpha, A_data, &SA0,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else {
        PyErr_SetString(PyExc_AssertionError,
                        "A is neither C nor F-contiguous, it should have been copied");
        return -1;
    }
    return 0;
}

/*
 * Call dgemv_ for double matrix-vector multiply.
 *
 * Handles both C-contiguous and F-contiguous layouts.
 * For C-contiguous (SA1 == 1): uses transpose
 * For F-contiguous (SA0 == 1): no transpose
 *
 * Returns 0 on success, -1 on error.
 */
static inline int pytensor_dgemv_dispatch(
    int NA0, int NA1,
    int SA0, int SA1,
    double *A_data, double *x_data, double *z_data,
    double alpha, double beta,
    int Sx, int Sz
) {
    if (SA0 == 1) {
        /* F-contiguous */
        char NOTRANS = 'N';
        dgemv_(&NOTRANS, &NA0, &NA1,
               &alpha, A_data, &SA1,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else if (SA1 == 1) {
        /* C-contiguous */
        char TRANS = 'T';
        dgemv_(&TRANS, &NA1, &NA0,
               &alpha, A_data, &SA0,
               x_data, &Sx,
               &beta, z_data, &Sz);
    } else {
        PyErr_SetString(PyExc_AssertionError,
                        "A is neither C nor F-contiguous, it should have been copied");
        return -1;
    }
    return 0;
}

/*
 * Handle vector-vector dot product case (when A has only 1 row).
 *
 * Computes: z[0] = beta * z[0] + alpha * dot(A[0,:], x)
 *
 * This is faster than calling gemv for a single row.
 */
static inline void pytensor_sgemv_dot_case(
    int NA1, int SA1,
    float *A_data, float *x_data, float *z_data,
    float alpha, float beta, int Sx
) {
    z_data[0] = (beta != 0.0f) ? beta * z_data[0] : 0.0f;
    z_data[0] += alpha * sdot_(&NA1, A_data, &SA1, x_data, &Sx);
}

static inline void pytensor_dgemv_dot_case(
    int NA1, int SA1,
    double *A_data, double *x_data, double *z_data,
    double alpha, double beta, int Sx
) {
    z_data[0] = (beta != 0.0) ? beta * z_data[0] : 0.0;
    z_data[0] += alpha * ddot_(&NA1, A_data, &SA1, x_data, &Sx);
}

#endif /* PYTENSOR_GEMV_HELPER_H */

"""
