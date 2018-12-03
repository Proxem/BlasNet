/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.BlasNet
{
    /// <summary>
    /// Wrapper for Intel's libmklml.so
    /// </summary>
    public unsafe class DynMklMlLinux: IBlas
    {
        const string MKLML_LINUX_DLL = "libmklml.so";
        
        private static Xerbla xerblaCallbackReferenceToStopItBeingGarbageCollected; // useful ?
        
        /// <param name="max">The max number of threads to use</param>
        public DynMklMlLinux(int max = -1)
        {
            xerblaCallbackReferenceToStopItBeingGarbageCollected = ErrorHandler;
            mkl_set_xerbla(xerblaCallbackReferenceToStopItBeingGarbageCollected);
            if (max > 0) SetNumThreads(max);
        }

        static void ErrorHandler(string name, int[] num, int len)
        {
            // https://software.intel.com/en-us/node/522122
            var info = num[0];
            switch (info)
            {
                case 1001:
                    throw new Exception(string.Format("Intel MKL ERROR: Incompatible optional parameters on entry to {0}.", name));
                case 1212:
                    throw new Exception(string.Format("Intel MKL INTERNAL ERROR: Issue accessing coprocessor in function {0}.", name));
                case 1000:
                case 1089:
                    throw new Exception(string.Format("Intel MKL INTERNAL ERROR: Insufficient workspace available in function {0}.", name));
                default:
                    if (info < 0)
                        throw new Exception(string.Format("Intel MKL INTERNAL ERROR: Condition {0} detected in function {1}.", -info, name));
                    else
                        throw new Exception(string.Format("Intel MKL ERROR: Parameter {0} was incorrect on entry to {1}.", info, name));
            }
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        delegate void Xerbla(string name, int[] num, int len);

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern IntPtr mkl_set_xerbla(Xerbla x);   // do not try to unmarshal return value => wrong appdomain exception

        /// <summary>
        /// preferred order for arrays, row or column major. 
        /// </summary>
        public Order PreferredOrder
        {
            get { return Order.RowMajor; }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void MKL_Set_Num_Threads(int n);

        /// <summary>
        /// Change the number of threads used by MKL.
        /// </summary>
        public void SetNumThreads(int n)
        {
            MKL_Set_Num_Threads(n);
        }

        /// <summary>
        /// returns the max number of threads that MKL can use.
        /// </summary>
        public int GetNumThreads()
        {
            return MKL_Get_Max_Threads();
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern int MKL_Get_Max_Threads();

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_dcopy(int n, double* x, int incx, double* y, int incy);

        /// <summary>
        /// copy x to y (double precision arrays)
        /// </summary>
        /// <param name="n"> number of elements to copy </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        /// <param name="y"> second array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void dcopy(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            fixed (double* xp = &x[offsetx], yp = &y[offsety])
            {
                cblas_dcopy(n, xp, incx, yp, incy);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_daxpy(int n, double a, double* x, int incx, double* y, int incy);

        /// <summary>
        /// performs y = y + ax (double precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> x's scaling factor </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        /// <param name="y"> second array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void daxpy(int n, double a, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            fixed (double* xp = &x[offsetx], yp = &y[offsety])
            {
                cblas_daxpy(n, a, xp, incx, yp, incy);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_dcscmm(int TransA, int m, int n, int k, double alpha, double* val, int* indx, int* pntrb, double* b, int ldb, double beta, double* c, int ldc);

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format. (double precision arrays)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void dcscmm(Transpose TransA, int m, int n, int k, double alpha, double[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc)
        {
            fixed (double* valp = &val[offsetval], bp = &b[offsetb], cp = &c[offsetc])
            fixed (int* indxp = &indx[offsetindx], pntrbp = &pntrb[offsetpntrb])
            {
                cblas_dcscmm((int)TransA, m, n, k, alpha, valp, indxp, pntrbp, bp, ldb, beta, cp, ldc);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_dcsrmm(int TransA, int m, int n, int k, double alpha, double* val, int* indx, int* pntrb, double* b, int ldb, double beta, double* c, int ldc);

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format. (double precision arrays)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void dcsrmm(Transpose TransA, int m, int n, int k, double alpha, double[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc)
        {
            fixed (double* valp = &val[offsetval], bp = &b[offsetb], cp = &c[offsetc])
            fixed (int* indxp = &indx[offsetindx], pntrbp = &pntrb[offsetpntrb])
            {
                cblas_dcsrmm((int)TransA, m, n, k, alpha, valp, indxp, pntrbp, bp, ldb, beta, cp, ldc);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern double cblas_ddot(int n, double* x, int incx, double* y, int incy);

        /// <summary>
        /// returns scalar product of x and y. (double precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        /// <param name="y"> second array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double ddot(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            fixed (double* xp = &x[offsetx], yp = &y[offsety])
            {
                return cblas_ddot(n, xp, incx, yp, incy);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_dgemm(int Order, int TransA, int TransB, int M, int N, int K, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc);

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c)
        /// </summary>
        /// <param name="Order">row or column major </param>
        /// <param name="TransA"> should A be transposed </param>
        /// <param name="TransB"> should B be transposed </param>
        /// <param name="M"> shape[0] A </param>
        /// <param name="N">shape [1] of B</param>
        /// <param name="K"> shape[1] of A and shape[0] of B</param>
        /// <param name="alpha"> scaling factor of AB</param>
        /// <param name="A"> first matrix </param>
        /// <param name="offseta"> starting offset in A </param>
        /// <param name="lda">leading dimension of A associated with Order </param>
        /// <param name="B"> second matrix </param>
        /// <param name="offsetb"> starting offset in B </param>
        /// <param name="ldb"> leading dimension of B associated with Order </param>
        /// <param name="beta"> scaling factor of C </param>
        /// <param name="C"> result matrix </param>
        /// <param name="offsetc"> starting offset in C </param>
        /// <param name="ldc"> leading dimension of C associated with Order </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void dgemm(Order Order, Transpose TransA, Transpose TransB, int M, int N, int K, double alpha, double[] A, int offseta, int lda, double[] B, int offsetb, int ldb, double beta, double[] C, int offsetc, int ldc)
        {
            fixed (double* Ap = &A[offseta], Bp = &B[offsetb], Cp = &C[offsetc])
            {
                cblas_dgemm((int)Order, (int)TransA, (int)TransB, M, N, K, alpha, Ap, lda, Bp, ldb, beta, Cp, ldc);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_dgemv(int Order, int trans, int m, int n, double alpha, double* a, int lda, double* x, int incx, double beta, double* y, int incy);

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y). (single precision arrays)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void dgemv(Order Order, Transpose trans, int m, int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy)
        {
            fixed (double* ap = &a[offseta], xp = &x[offsetx], yp = &y[offsety])
            {
                cblas_dgemv((int)Order, (int)trans, m, n, alpha, ap, lda, xp, incx, beta, yp, incy);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_dscal(int n, double a, double* x, int incx);

        /// <summary>
        /// multiply selected elements of x by a. (double precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> scaling parameter </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void dscal(int n, double a, double[] x, int offsetx, int incx)
        {
            fixed (double* xp = &x[offsetx])
            {
                cblas_dscal(n, a, xp, incx);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_scopy(int n, float* x, int incx, float* y, int incy);

        /// <summary>
        /// copy x in y. (single precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        /// <param name="y"> second array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void scopy(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            fixed (float* xp = &x[offsetx], yp = &y[offsety])
            {
                cblas_scopy(n, xp, incx, yp, incy);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_saxpy(int n, float a, float* x, int incx, float* y, int incy);

        /// <summary>
        /// performs y = ax + y. (single precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"></param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        /// <param name="y"> second array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void saxpy(int n, float a, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            fixed (float* xp = &x[offsetx], yp = &y[offsety])
            {
                cblas_saxpy(n, a, xp, incx, yp, incy);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_scscmm(int TransA, int m, int n, int k, float alpha, float* val, int* indx, int* pntrb, float* b, int ldb, float beta, float* c, int ldc);

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format. (single precision arrays)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void scscmm(Transpose TransA, int m, int n, int k, float alpha, float[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc)
        {
            fixed (float* valp = &val[offsetval], bp = &b[offsetb], cp = &c[offsetc])
            fixed (int* indxp = &indx[offsetindx], pntrbp = &pntrb[offsetpntrb])
            {
                cblas_scscmm((int)TransA, m, n, k, alpha, valp, indxp, pntrbp, bp, ldb, beta, cp, ldc);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void mkl_scsrmm(byte* TransA, int* m, int* n, int* k, float* alpha, byte* descra, float* val, int* indx, int* pntrb, int* pntre, float* b, int* ldb, float* beta, float* c, int* ldc);

        /// <summary>
        /// Default description of a matrice. If the matrix is diagonal or triangular consider using a more approriate description
        /// </summary>
        static readonly byte[] DEFAULT = new byte[] { (byte)'G', 0, 0, (byte)'C', 0, 0 };

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format. (single precision arrays)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void scsrmm(Transpose TransA, int m, int n, int k, float alpha, float[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc)
        {
            var trans = (byte)(TransA == Transpose.Trans ? 'T' : TransA == Transpose.ConjTrans ? 'C' : 'N');

            fixed (float* valp = &val[offsetval], bp = &b[offsetb], cp = &c[offsetc])
            fixed (int* indxp = &indx[offsetindx], pntrbp = &pntrb[offsetpntrb])
            fixed (byte* descrap = &DEFAULT[0])
            {
                mkl_scsrmm(&trans, &m, &n, &k, &alpha, descrap, valp, indxp, pntrbp, pntrbp + 1, bp, &ldb, &beta, cp, &ldc);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern float cblas_sdot(int n, float* x, int incx, float* y, int incy);

        /// <summary>
        /// return scalar product of x and y. (single precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        /// <param name="y"> second array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float sdot(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            fixed (float* xp = &x[offsetx], yp = &y[offsety])
            {
                return cblas_sdot(n, xp, incx, yp, incy);
                //float result2 = 0;
                //var xp2 = xp;
                //var yp2 = yp;
                //for (int i = 0; i < n; i++)
                //{
                //    result2 += *xp2 * *yp2;
                //    xp2 += incx;
                //    yp2 += incy;
                //}
                //return result2;
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc);

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c) (single precision arrays)
        /// </summary>
        /// <param name="Order">row or column major </param>
        /// <param name="TransA"> should A be transposed </param>
        /// <param name="TransB"> should B be transposed </param>
        /// <param name="M"> shape[0] A </param>
        /// <param name="N">shape [1] of B</param>
        /// <param name="K"> shape[1] of A and shape[0] of B</param>
        /// <param name="alpha"> scaling factor of AB</param>
        /// <param name="A"> first matrix </param>
        /// <param name="offseta"> starting offset in A </param>
        /// <param name="lda">leading dimension of A associated with Order </param>
        /// <param name="B"> second matrix </param>
        /// <param name="offsetb"> starting offset in B </param>
        /// <param name="ldb"> leading dimension of B associated with Order </param>
        /// <param name="beta"> scaling factor of C </param>
        /// <param name="C"> result matrix </param>
        /// <param name="offsetc"> starting offset in C </param>
        /// <param name="ldc"> leading dimension of C associated with Order </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void sgemm(Order Order, Transpose TransA, Transpose TransB, int M, int N, int K, float alpha, float[] A, int offseta, int lda, float[] B, int offsetb, int ldb, float beta, float[] C, int offsetc, int ldc)
        {
            fixed (float* Ap = &A[offseta], Bp = &B[offsetb], Cp = &C[offsetc])
            {
                cblas_sgemm((int)Order, (int)TransA, (int)TransB, M, N, K, alpha, Ap, lda, Bp, ldb, beta, Cp, ldc);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_sgemv(int Order, int trans, int m, int n, float alpha, float* a, int lda, float* x, int incx, float beta, float* y, int incy);

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y). (single precision arrays)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void sgemv(Order Order, Transpose trans, int m, int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy)
        {
            fixed (float* ap = &a[offseta], xp = &x[offsetx], yp = &y[offsety])
            {
                cblas_sgemv((int)Order, (int)trans, m, n, alpha, ap, lda, xp, incx, beta, yp, incy);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void cblas_sscal(int n, float a, float* x, int incx);

        /// <summary>
        /// multiply selected elements of x by a. (single precision array)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> scaling factor </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void sscal(int n, float a, float[] x, int offsetx, int incx)
        {
            fixed (float* xp = &x[offsetx])
            {
                cblas_sscal(n, a, xp, incx);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vdAdd(int n, double* a, double* b, double* r);

        /// <summary>
        /// fill r with a + b. (double precision array)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vdadd(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            fixed (double* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vdAdd(n, ap, bp, rp);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vdSub(int n, double* a, double* b, double* r);

        /// <summary>
        /// fill r with a - b. (double precision array)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vdsub(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            fixed (double* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vdSub(n, ap, bp, rp);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vdDiv(int n, double* a, double* b, double* r);

        /// <summary>
        /// fill r with a / b. (double precision array)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vddiv(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            fixed (double* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vdDiv(n, ap, bp, rp);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vdMul(int n, double* a, double* b, double* r);

        /// <summary>
        /// fill r with a * b. (double precision array)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vdmul(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            fixed (double* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vdMul(n, ap, bp, rp);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vsAdd(int n, float* a, float* b, float* r);

        /// <summary>
        /// fill r with a + b. (single precision array) 
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vsadd(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            fixed (float* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vsAdd(n, ap, bp, rp);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vsSub(int n, float* a, float* b, float* r);

        /// <summary>
        /// fill r with a - b. (single precision array) 
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vssub(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            fixed (float* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vsSub(n, ap, bp, rp);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vsDiv(int n, float* a, float* b, float* r);

        /// <summary>
        /// fill r with a / b. (single precision array) 
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vsdiv(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            fixed (float* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vsDiv(n, ap, bp, rp);
            }
        }

        [DllImport(MKLML_LINUX_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        static extern void vsMul(int n, float* a, float* b, float* r);

        /// <summary>
        /// fill r with a * b. (single precision array) 
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> first array </param>
        /// <param name="offseta"> starting offset in a </param>
        /// <param name="b"> second array </param>
        /// <param name="offsetb">  starting offset in b </param>
        /// <param name="r"> receiving array </param>
        /// <param name="offsetr">  starting offset in r </param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void vsmul(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            fixed (float* ap = &a[offseta], bp = &b[offsetb], rp = &r[offsetr])
            {
                vsMul(n, ap, bp, rp);
            }
        }

        internal static void Test(int LOOP_COUNT, int m, int p, int n)
        {
            Blas.Provider = new DynMklMlLinux(-1);

            double[] A, B, C;
            int /*m, n, p,*/ i, j, r, max_threads;
            double alpha, beta;
            var s_initial = new System.Diagnostics.Stopwatch();
            long s_elapsed = 0;

            Console.Write("\n This example demonstrates threading impact on computing real matrix product \n" +
                " C=alpha*A*B+beta*C using Intel(R) MKL function dgemm, where A, B, and C are \n" +
                " matrices and alpha and beta are double precision scalars \n\n");

            //m = 2000, p = 200, n = 1000;
            Console.Write(" Initializing data for matrix multiplication C=A*B for matrix \n" +
                " A({0}x{1}) and matrix B({2}x{3})\n\n", m, p, p, n);
            alpha = 1.0; beta = 0.0;

            Console.Write(" Allocating memory for matrices aligned on 64-byte boundary for better \n" +
                " performance \n\n");
            A = new double[m * p];
            B = new double[p * n];
            C = new double[m * n];

            Console.Write(" Intializing matrix data \n\n");
            for (i = 0; i < (m * p); i++)
            {
                A[i] = (double)(i + 1);
            }

            for (i = 0; i < (p * n); i++)
            {
                B[i] = (double)(-i - 1);
            }

            for (i = 0; i < (m * n); i++)
            {
                C[i] = 0.0;
            }

            Console.Write(" Finding max number of threads Intel(R) MKL can use for parallel runs \n\n");
            max_threads = MKL_Get_Max_Threads();

            Console.Write(" Running Intel(R) MKL from 1 to {0} threads \n\n", max_threads);
            for (i = 1; i <= max_threads; i++)
            {
                for (j = 0; j < (m * n); j++)
                    C[j] = 0.0;

                Console.Write(" Requesting Intel(R) MKL to use {0} thread(s) \n\n", i);
                Blas.Provider.SetNumThreads(i);

                Console.Write(" Making the first run of matrix product using Intel(R) MKL dgemm function \n" +
                    " via CBLAS interface to get stable run time measurements \n\n");
                Blas.gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans,
                    m, n, p, alpha, A, 0, p, B, 0, n, beta, C, 0, n);

                Console.Write(" Measuring performance of matrix product using Intel(R) MKL dgemm function \n" +
                    " via CBLAS interface on {0} thread(s) \n\n", i);
                s_initial.Start();
                for (r = 0; r < LOOP_COUNT; r++)
                {
                    Blas.gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans,
                        m, n, p, alpha, A, 0, p, B, 0, n, beta, C, 0, n);
                }
                s_elapsed = s_initial.ElapsedMilliseconds / LOOP_COUNT;

                Console.Write(" == Matrix multiplication using Intel(R) MKL dgemm completed ==\n" +
                    " == at {0:F5} milliseconds using {1} thread(s) ==\n\n", s_elapsed, i);
            }

            if (s_elapsed < 0.9 / LOOP_COUNT)
            {
                s_elapsed = (long)(1.0 / LOOP_COUNT / s_elapsed);
                i = (int)(s_elapsed * LOOP_COUNT) + 1;
                Console.Write(" It is highly recommended to define LOOP_COUNT for this example on your \n" +
                    " computer as {0} to have total execution time about 1 second for reliability \n" +
                    " of measurements\n\n", i);
            }

            Console.Write(" Example completed. \n\n");
        }
    }
}
