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
using System.Text;
using System.Threading.Tasks;

namespace Proxem.BlasNet
{
    /// <summary>
    /// C# implementation of the basic array function (way slower than MKL)
    /// </summary>
    public class DefaultBlas : IBlas
    {
        /// <summary>
        /// preferred order for matrices, row or column major
        /// </summary>
        public Order PreferredOrder => Order.RowMajor;

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
        public void daxpy(int n, double a, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            for(int i = 0; i < n; ++i)
            {
                y[offsety] += a * x[offsetx];
                offsetx += incx;
                offsety += incy;
            }
        }

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
        public void saxpy(int n, float a, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            for (int i = 0; i < n; ++i)
            {
                y[offsety] += a * x[offsetx];
                offsetx += incx;
                offsety += incy;
            }
        }

        /// <summary>
        /// performs y = ax + y. (single precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"></param>
        /// <param name="x"> value to add to y </param>
        /// <param name="y"> array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        public void saxpy(int n, float a, ref float x, float[] y, int offsety, int incy)
        {
            for (int i = 0; i < n; ++i)
            {
                y[offsety] += a * x;
                offsety += incy;
            }
        }

        /// <summary>
        /// performs y = ax + y. (double precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"></param>
        /// <param name="x"> value to add to y </param>
        /// <param name="y"> array </param>
        /// <param name="offsety"> starting offset in receiving array y</param>
        /// <param name="incy"> increment in receiving array </param>
        public void daxpy(int n, double a, ref double x, double[] y, int offsety, int incy)
        {
            for (int i = 0; i < n; ++i)
            {
                y[offsety] += a * x;
                offsety += incy;
            }
        }

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
        public void dcopy(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            for (int i = 0; i < n; ++i)
            {
                y[offsety] = x[offsetx];
                offsetx += incx;
                offsety += incy;
            }
        }

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format. (double precision arrays)
        /// </summary>
        public void dcscmm(Transpose TransA, int m, int n, int k, double alpha, double[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format. (double precision arrays)
        /// </summary>
        public void dcsrmm(Transpose TransA, int m, int n, int k, double alpha, double[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, double[] b, int offsetb, int ldb, double beta, double[] c, int offsetc, int ldc)
        {
            throw new NotImplementedException();
        }

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
        public double ddot(int n, double[] x, int offsetx, int incx, double[] y, int offsety, int incy)
        {
            double dot = 0;
            for (int i = 0; i < n; ++i)
            {
                dot += y[offsety] * x[offsetx];
                offsetx += incx;
                offsety += incy;
            }
            return dot;
        }

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c)
        /// </summary>
        /// <param name="Order">row or column major </param>
        /// <param name="TransA"> should A be transposed </param>
        /// <param name="TransB"> should B be transposed </param>
        /// <param name="M"> shape[0] A </param>
        /// <param name="N"> shape[1] of A and shape[0] of B</param>
        /// <param name="K">shape [1] of B</param>
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
        public void dgemm(Order Order, Transpose TransA, Transpose TransB, int M, int N, int K, double alpha, double[] A, int offseta, int lda, double[] B, int offsetb, int ldb, double beta, double[] C, int offsetc, int ldc)
        {
            var incColA = Order == Order.ColMajor ^ TransA != Transpose.NoTrans;
            int inca_i = incColA ? 1 : lda;
            int inca_k = incColA ? lda : 1;

            var incColB = Order == Order.ColMajor ^ TransB != Transpose.NoTrans;
            int incb_k = incColB ? 1 : ldb;
            int incb_j = incColB ? ldb : 1;

            var incColC = Order == Order.ColMajor;
            int incc_i = incColC ? 1 : ldc;
            int incc_j = incColC ? ldc : 1;

            int offa_i = offseta, offa_ik;
            int offb_k = offsetb, offb_kj;
            int offc_i = offsetc, offc_ij;

            // reset C
            for (int i = 0; i < M; ++i)
            {
                offc_ij = offc_i;
                for (int j = 0; j < N; ++j)
                {
                    C[offc_ij] = beta * C[offc_ij];
                    offc_ij += incc_j;
                }
                offc_i += incc_i;
            }
            offc_i = offsetc;

            // compute the matrix multiplication
            for (int i = 0; i < M; ++i)
            {
                offa_ik = offa_i;
                offb_k = offsetb;
                for (int k = 0; k < K; ++k)
                {
                    var r = A[offa_ik];
                    offc_ij = offc_i;
                    offb_kj = offb_k;
                    for (int j = 0; j < N; ++j)
                    {
                        C[offc_ij] += alpha * r * B[offb_kj];
                        offb_kj += incb_j;
                        offc_ij += incc_j;
                    }
                    offa_ik += inca_k;
                    offb_k += incb_k;
                }
                offa_i += inca_i;
                offc_i += incc_i;
            }
        }

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y). (single precision arrays)
        /// </summary>
        public void dgemv(Order Order, Transpose trans, int m, int n, double alpha, double[] a, int offseta, int lda, double[] x, int offsetx, int incx, double beta, double[] y, int offsety, int incy)
        {
            var incCol = Order == Order.ColMajor ^ trans != Transpose.NoTrans;
            int inca_i = incCol ? 1 : lda;
            int inca_j = incCol ? lda : 1;

            if (trans != Transpose.NoTrans)
            {
                int mm = m;
                m = n;
                n = mm;
            }

            if (beta != 1)
                dscal(m, beta, y, offsety, incy);

            int offa_i = offseta, offa_ij;
            for (int i = 0; i < m; ++i)
            {
                offa_ij = offa_i;
                var offx = offsetx;
                for (int j = 0; j < n; ++j)
                {
                    y[offsety] += alpha * a[offa_ij] * x[offx];
                    offa_ij += inca_j;
                    offx += incx;
                }
                offsety += incy;
                offa_i += inca_i;
            }
        }

        /// <summary>
        /// multiply selected elements of x by a. (double precision arrays)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> scaling parameter </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        public void dscal(int n, double a, double[] x, int offsetx, int incx)
        {
            for (int i = 0; i < n; ++i)
            {
                x[offsetx] *= a;
                offsetx += incx;
            }
        }

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
        public void scopy(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            for (int i = 0; i < n; ++i)
            {
                y[offsety] = x[offsetx];
                offsetx += incx;
                offsety += incy;
            }
        }

        /// <summary>
        /// Computes matrix-matrix product of a sparse matrix stored in the CSC format. (single precision arrays)
        /// </summary>
        public void scscmm(Transpose TransA, int m, int n, int k, float alpha, float[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, float[] b, int offsetb, int ldb, float beta, float[] c, int offsetc, int ldc)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Computes matrix - matrix product of a sparse matrix stored in the CSR format. (single precision arrays)
        /// </summary>
        public void scsrmm(Transpose TransA, int M, int N, int K, float alpha, float[] val, int offsetval, int[] indx, int offsetindx, int[] pntrb, int offsetpntrb, float[] B, int offsetb, int ldb, float beta, float[] C, int offsetc, int ldc)
        {
            var incColB = false;
            int incb_k = incColB ? 1 : ldb;
            int incb_j = incColB ? ldb : 1;

            int incc_i = ldc;
            int incc_j = 1;

            int offa_i = offsetval;
            int offb_j = offsetb;
            int offc_i = offsetc, offc_ij;

            // reset C
            for (int i = 0; i < M; ++i)
            {
                offc_ij = offc_i;
                for (int j = 0; j < N; ++j)
                {
                    C[offc_ij] = beta * C[offc_ij];
                    offc_ij += incc_j;
                }
                offc_i += incc_i;
            }
            offc_i = offsetc;

            int offb_k = offsetb, offb_kj;

            // compute the matrix multiplication
            for (int i = 0; i < M; ++i)
            {
                while (offsetval < pntrb[offsetpntrb + 1])
                {
                    var r = val[offsetval];
                    offc_ij = offc_i;
                    offb_k = offsetb + indx[offsetindx] * incb_k;
                    offb_kj = offb_k ;
                    for (int j = 0; j < N; ++j)
                    {
                        C[offc_ij] += alpha * r * B[offb_kj];
                        offb_kj += incb_j;
                        offc_ij += incc_j;
                    }
                    ++offsetval;
                    ++offsetindx;
                }
                ++offsetpntrb;
                offc_i += incc_i;
            }
        }
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
        /// <returns></returns>
        public float sdot(int n, float[] x, int offsetx, int incx, float[] y, int offsety, int incy)
        {
            float dot = 0;
            for (int i = 0; i < n; ++i)
            {
                dot += y[offsety] * x[offsetx];
                offsetx += incx;
                offsety += incy;
            }
            return dot;
        }

        /// <summary>
        /// Computes a scalar-matrix-matrix product and adds the result to a scalar-matrix product. (c := alpha*op(A)*op(b) + beta*c) (single precision arrays)
        /// </summary>
        /// <param name="Order">row or column major </param>
        /// <param name="TransA"> should A be transposed </param>
        /// <param name="TransB"> should B be transposed </param>
        /// <param name="M"> shape[0] A </param>
        /// <param name="N"> shape[1] of A and shape[0] of B</param>
        /// <param name="K">shape [1] of B</param>
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
        public void sgemm(Order Order, Transpose TransA, Transpose TransB, int M, int N, int K, float alpha, float[] A, int offseta, int lda, float[] B, int offsetb, int ldb, float beta, float[] C, int offsetc, int ldc)
        {
            var incColA = Order == Order.ColMajor ^ TransA != Transpose.NoTrans;
            int inca_i = incColA ? 1 : lda;
            int inca_k = incColA ? lda : 1;

            var incColB = Order == Order.ColMajor ^ TransB != Transpose.NoTrans;
            int incb_k = incColB ? 1 : ldb;
            int incb_j = incColB ? ldb : 1;

            var incColC = Order == Order.ColMajor;
            int incc_i = incColC ? 1 : ldc;
            int incc_j = incColC ? ldc : 1;

            int offa_i = offseta, offa_ik;
            int offb_k = offsetb, offb_kj;
            int offc_i = offsetc, offc_ij;

            // reset C
            for (int i = 0; i < M; ++i)
            {
                offc_ij = offc_i;
                for (int j = 0; j < N; ++j)
                {
                    C[offc_ij] = beta * C[offc_ij];
                    offc_ij += incc_j;
                }
                offc_i += incc_i;
            }
            offc_i = offsetc;

            // compute the matrix multiplication
            for (int i = 0; i < M; ++i)
            {
                offa_ik = offa_i;
                offb_k = offsetb;
                for (int k = 0; k < K; ++k)
                {
                    var r = A[offa_ik];
                    offc_ij = offc_i;
                    offb_kj = offb_k;
                    for (int j = 0; j < N; ++j)
                    {
                        C[offc_ij] += alpha * r * B[offb_kj];
                        offb_kj += incb_j;
                        offc_ij += incc_j;
                    }
                    offa_ik += inca_k;
                    offb_k += incb_k;
                }
                offa_i += inca_i;
                offc_i += incc_i;
            }
        }

        /// <summary>
        /// Computes a matrix-vector product using a general matrix. (y := alpha*A*x + beta*y). (single precision arrays)
        /// </summary>
        public void sgemv(Order Order, Transpose trans, int m, int n, float alpha, float[] a, int offseta, int lda, float[] x, int offsetx, int incx, float beta, float[] y, int offsety, int incy)
        {
            var incCol = Order == Order.ColMajor ^ trans != Transpose.NoTrans;
            int inca_i = incCol ? 1 : lda;
            int inca_j = incCol ? lda : 1;

            if (trans != Transpose.NoTrans)
            {
                int mm = m;
                m = n;
                n = mm;
            }

            if (beta != 1)
                sscal(m, beta, y, offsety, incy);

            int offa_i = offseta, offa_ij;
            for(int i = 0; i < m; ++i)
            {
                offa_ij = offa_i;
                var offx = offsetx;
                for (int j = 0; j < n; ++j)
                {
                    y[offsety] += alpha * a[offa_ij] * x[offx];
                    offa_ij += inca_j;
                    offx += incx;
                }
                offsety += incy;
                offa_i += inca_i;
            }
        }

        /// <summary>
        /// multiply selected elements of x by a. (single precision array)
        /// </summary>
        /// <param name="n"> number of elementary operations to perform </param>
        /// <param name="a"> scaling factor </param>
        /// <param name="x"> first array </param>
        /// <param name="offsetx"> starting offset in copied array x </param>
        /// <param name="incx"> increment in copied array </param>
        public void sscal(int n, float a, float[] x, int offsetx, int incx)
        {
            for (int i = 0; i < n; ++i)
            {
                x[offsetx] *= a;
                offsetx += incx;
            }
        }

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
        public void vdadd(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] + b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

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
        public void vdsub(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] - b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

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
        public void vddiv(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] / b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

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
        public void vdmul(int n, double[] a, int offseta, double[] b, int offsetb, double[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] * b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

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
        public void vsadd(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] + b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

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
        public void vssub(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] - b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

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
        public void vsdiv(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] / b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

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
        public void vsmul(int n, float[] a, int offseta, float[] b, int offsetb, float[] r, int offsetr)
        {
            for (int i = 0; i < n; ++i)
            {
                r[offsetr] = a[offseta] * b[offsetb];
                offseta += 1;
                offsetb += 1;
                offsetr += 1;
            }
        }

        /// <summary>
        /// retrieve max number of threads used by Acml.
        /// </summary>
        /// <returns></returns>
        public int GetNumThreads()
        {
            return 1;
        }

        /// <summary>
        /// set max number of threads used by Acml.
        /// </summary>
        /// <param name="n">max number of threads</param>
        public void SetNumThreads(int n)
        {
            return;
        }
    }
}
