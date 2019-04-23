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
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

namespace Proxem.BlasNet
{
#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member
    /// <summary>
    /// Convention for indexing matrices, row or column Major.
    /// In row major (C), M[i, j] is the element at the i-row and j-column of M.
    /// In column major (Fortran), M[i, j] is the element at the i-column and j-row of M.
    /// </summary>
    public enum Order
    {
        RowMajor = 101, ColMajor = 102
    }

    /// <summary>
    /// Used to indicates if a matrix needs to be transposed before doing the calculation.
    /// For performance reason,
    /// Blas implementations usually don't actually transpose the matrix but modify their algorithm.
    /// </summary>
    public enum Transpose
    {
        NoTrans = 111, Trans = 112, ConjTrans = 113
    }
#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member

    /// <summary>
    /// Blas stands for Basic Linear Algebra Subprogram.
    /// <see href="http://http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms">Blas</see>
    /// is a standard defining common operations to work with vectors and matrices.
    /// The operations include +, -, *, / and dot product.
    ///
    /// This interface allows to wrap different Blas implementations.
    /// The mkl implementation of Blas has been wrapped in <see cref="DynMkl"/>
    /// </summary>
    public interface IBlas
    {
        /// <summary>
        /// The prefered convention for matrices, Row or Col Major.
        /// <see cref="Order"/>
        /// Some implementations are faster for one convention.
        /// </summary>
        Order PreferredOrder { get; }

        /// <summary>
        /// Copies a vector into another vector
        /// </summary>
        void scopy(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy);

        /// <summary>
        /// Copies a vector into another vector
        /// </summary>
        void dcopy(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy);

        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        void saxpy(int n, float a, float[] x, int offsetx, int incx, float[] y, int offsety, int incy);

        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        void daxpy(int n, double a, double[] x, int offsetx, int incx, double[] y, int offsety, int incy);

        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        void saxpy(int n, float a, ref float x, float[] y, int offsety, int incy);

        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        void daxpy(int n, double a, ref double x, double[] y, int offsety, int incy);

        /// <summary>
        /// Computes the product of a vector by a scalar. (x = a*x)
        /// </summary>
        void sscal(int n, float a, float[] x, int offsetx, int incx);

        /// <summary>
        /// Computes the product of a vector by a scalar. (x = a*x)
        /// </summary>
        void dscal(int n, double a, double[] x, int offsetx, int incx);

        /// <summary>
        /// forms the dot product of two vectors.
        /// </summary>
        float sdot(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy);

        /// <summary>
        /// forms the dot product of two vectors.
        /// </summary>
        double ddot(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy);

        /// <summary>
        /// Addition: r[i] = a[i] + b[i]
        /// </summary>
        void vsadd(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr);

        /// <summary>
        /// Addition: r[i] = a[i] + b[i]
        /// </summary>
        void vdadd(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr);

        /// <summary>
        /// Substraction: r[i] = a[i] - b[i]
        /// </summary>
        void vssub(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr);

        /// <summary>
        /// Substraction: r[i] = a[i] - b[i]
        /// </summary>
        void vdsub(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr);

        /// <summary>
        /// Multiplication: r[i] = a[i] * b[i]
        /// </summary>
        void vsmul(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr);

        /// <summary>
        /// Multiplication: r[i] = a[i] * b[i]
        /// </summary>
        void vdmul(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr);

        /// <summary>
        /// Division: r[i] = a[i] / b[i]
        /// </summary>
        void vsdiv(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr);

        /// <summary>
        /// Division: r[i] = a[i] / b[i]
        /// </summary>
        void vddiv(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr);

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y)
        /// </summary>
        void sgemv(Order Order, Transpose trans,
            int m, int n, float alpha,
            float[] a, int offseta, int lda,
            float[] x, int offsetx, int incx,
            float beta,
            float[] y, int offsety, int incy);

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y)
        /// </summary>
        void dgemv(Order Order, Transpose trans,
            int m, int n, double alpha,
            double[] a, int offseta, int lda,
            double[] x, int offsetx, int incx,
            double beta,
            double[] y, int offsety, int incy);

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c)
        /// </summary>
        void sgemm(Order Order, Transpose TransA,
            Transpose TransB, int M, int N,
            int K, float alpha,
            float[] A, int offseta, int lda,
            float[] B, int offsetb, int ldb,
            float beta,
            float[] C, int offsetc, int ldc);

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c)
        /// </summary>
        void dgemm(Order Order, Transpose TransA,
            Transpose TransB, int M, int N,
            int K, double alpha,
            double[] A, int offseta, int lda,
            double[] B, int offsetb, int ldb,
            double beta,
            double[] C, int offsetc, int ldc);

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format.
        /// </summary>
        void scsrmm(Transpose TransA, int m, int n, int k,
            float alpha,
            float[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
            //int[] pntre, int offsetpntre,
            float[] b, int offsetb, int ldb,
            float beta,
            float[] c, int offsetc, int ldc);

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format.
        /// </summary>
        void dcsrmm(Transpose TransA, int m, int n, int k,
            double alpha,
            double[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
                //int[] pntre, int offsetpntre,
                double[] b, int offsetb, int ldb,
            double beta,
            double[] c, int offsetc, int ldc);

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format.
        /// </summary>
        void scscmm(Transpose TransA, int m, int n, int k,
            float alpha,
            float[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
            //int[] pntre, int offsetpntre,
            float[] b, int offsetb, int ldb,
            float beta,
            float[] c, int offsetc, int ldc);

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format.
        /// </summary>
        void dcscmm(Transpose TransA, int m, int n, int k,
            double alpha,
            double[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
            //int[] pntre, int offsetpntre,
            double[] b, int offsetb, int ldb,
            double beta,
            double[] c, int offsetc, int ldc);

        /// <summary>
        /// return max number of threads used by provider.
        /// </summary>
        int GetNumThreads();

        /// <summary>
        /// set max number of threads that provider can use.
        /// </summary>
        void SetNumThreads(int n);
    }

    /// <summary>
    /// Static wrapper around an <see cref="IBlas"> implementation </see>.
    /// All methods of NumNet calling Blas will use this facade.
    /// </summary>
    public static class Blas
    {
        /// <summary> The actual provider. </summary>
        public static IBlas Provider = new DefaultBlas();

        private static int _nthreads = Environment.ProcessorCount / 2;

        /// <summary> Number of CPU threads used. </summary>
        public static int NThreads
        {
            get { return _nthreads;  }
            set { if(value > 0) _nthreads = value; }
        }

        /// <summary>
        /// Preferred convention for matrices, row or column major.
        /// </summary>
        public static Order PreferredOrder
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get { return Provider.PreferredOrder; }
        }

        /// <summary>
        /// Copies a vector into another vector
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void copy(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            Provider.scopy(n, x, offsetx, incx, y, offsety, incy);
        }

        /// <summary>
        /// Copies a vector into another vector
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void copy(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            Provider.dcopy(n, x, offsetx, incx, y, offsety, incy);
        }

        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void axpy(int n, float a, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            Provider.saxpy(n, a, x, offsetx, incx, y, offsety, incy);
        }

        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void axpy(int n, double a, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            Provider.daxpy(n, a, x, offsetx, incx, y, offsety, incy);
        }

        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void axpy(int n, float a, ref float x, float[] y, int offsety, int incy)
        {
            Provider.saxpy(n, a, ref x, y, offsety, incy);
        }
        
        /// <summary>
        /// Computes a vector-scalar product and adds the result to a vector. (y := a*x + y)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void axpy(int n, double a, ref double x, double[] y, int offsety, int incy)
        {
            Provider.daxpy(n, a, ref x, y, offsety, incy);
        }


        /// <summary>
        /// Computes the product of a vector by a scalar. (x = a*x)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void scal(int n, float a, float[] x, int offsetx, int incx)
        {
            Provider.sscal(n, a, x, offsetx, incx);
        }

        /// <summary>
        /// Computes the product of a vector by a scalar. (x = a*x)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void scal(int n, double a, double[] x, int offsetx, int incx)
        {
            Provider.dscal(n, a, x, offsetx, incx);
        }

        /// <summary>
        /// forms the dot product of two vectors.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float dot(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            return Provider.sdot(n, x, offsetx, incx, y, offsety, incy);
        }

        /// <summary>
        /// forms the dot product of two vectors.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double dot(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            return Provider.ddot(n, x, offsetx, incx, y, offsety, incy);
        }

        /// <summary>
        /// Addition: r[i] = a[i] + b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vadd(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            Provider.vsadd(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Addition: r[i] = a[i] + b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vadd(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            Provider.vdadd(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Substraction: r[i] = a[i] - b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vsub(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            Provider.vssub(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Substraction: r[i] = a[i] - b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vsub(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            Provider.vdsub(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Multiplication: r[i] = a[i] * b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vmul(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            Provider.vsmul(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Multiplication: r[i] = a[i] * b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vmul(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            Provider.vdmul(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Division: r[i] = a[i] / b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vdiv(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            Provider.vsdiv(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Division: r[i] = a[i] / b[i]
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void vdiv(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            Provider.vddiv(n, a, offseta, b, offsetb, r, offsetr);
        }

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gemv(Order Order, Transpose trans,
            int m, int n, float alpha,
            float[] a, int offseta, int lda,
            float[] x, int offsetx, int incx,
            float beta,
            float[] y, int offsety, int incy)
        {
            Provider.sgemv(Order, trans, m, n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
        }

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gemv(Order Order, Transpose trans,
            int m, int n, double alpha,
            double[] a, int offseta, int lda,
            double[] x, int offsetx, int incx,
            double beta,
            double[] y, int offsety, int incy)
        {
            Provider.dgemv(Order, trans, m, n, alpha, a, offseta, lda, x, offsetx, incx, beta, y, offsety, incy);
        }

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gemm(Order Order, Transpose TransA,
            Transpose TransB, int M, int N,
            int K, float alpha,
            float[] A, int offseta, int lda,
            float[] B, int offsetb, int ldb,
            float beta,
            float[] C, int offsetc, int ldc)
        {
            Provider.sgemm(Order, TransA, TransB, M, N, K, alpha, A, offseta, lda, B, offsetb, ldb, beta, C, offsetc, ldc);
        }

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gemm(Order Order, Transpose TransA,
            Transpose TransB, int M, int N,
            int K, double alpha,
            double[] A, int offseta, int lda,
            double[] B, int offsetb, int ldb,
            double beta,
            double[] C, int offsetc, int ldc)
        {
            Provider.dgemm(Order, TransA, TransB, M, N, K, alpha, A, offseta, lda, B, offsetb, ldb, beta, C, offsetc, ldc);
        }

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void csrmm(Transpose TransA, int m, int n, int k,
            float alpha,
            float[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
            //int[] pntre, int offsetpntre,
            float[] b, int offsetb, int ldb,
            float beta,
            float[] c, int offsetc, int ldc)
        {
            Provider.scsrmm(TransA, m, n, k, alpha, val, offsetval, indx, offsetindx, pntrb, offsetpntrb, b, offsetb, ldb, beta, c, offsetc, ldc);
        }

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void csrmm(Transpose TransA, int m, int n, int k,
            double alpha,
            double[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
            //int[] pntre, int offsetpntre,
            double[] b, int offsetb, int ldb,
            double beta,
            double[] c, int offsetc, int ldc)
        {
            Provider.dcsrmm(TransA, m, n, k, alpha, val, offsetval, indx, offsetindx, pntrb, offsetpntrb, b, offsetb, ldb, beta, c, offsetc, ldc);
        }

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void cscmm(Transpose TransA, int m, int n, int k,
            float alpha,
            float[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
            //int[] pntre, int offsetpntre,
            float[] b, int offsetb, int ldb,
            float beta,
            float[] c, int offsetc, int ldc)
        {
            Provider.scscmm(TransA, m, n, k, alpha, val, offsetval, indx, offsetindx, pntrb, offsetpntrb, b, offsetb, ldb, beta, c, offsetc, ldc);
        }

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void cscmm(Transpose TransA, int m, int n, int k,
            double alpha,
            double[] val, int offsetval,
            int[] indx, int offsetindx,
            int[] pntrb, int offsetpntrb,
            //int[] pntre, int offsetpntre,
            double[] b, int offsetb, int ldb,
            double beta,
            double[] c, int offsetc, int ldc)
        {
            Provider.dcscmm(TransA, m, n, k, alpha, val, offsetval, indx, offsetindx, pntrb, offsetpntrb, b, offsetb, ldb, beta, c, offsetc, ldc);
        }
    }
}
