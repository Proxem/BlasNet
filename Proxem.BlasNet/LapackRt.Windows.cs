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
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.BlasNet
{
    /// <summary>
    /// wrapper for Intel's MKL high level functions on arrays.
    /// </summary>
    public unsafe class LapackRtWindows: ILapack
    {
        const string MKLRT_WINDOWS_DLL = "mkl_rt.dll";

        internal const int LAPACK_ROW_MAJOR = 101;

        internal const int LAPACK_COL_MAJOR = 102;

        ////////////////////////////////////
        //// Linear systems solution   /////
        //////////////////////////////////// 

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sspsv(int matrix_layout, char uplo, int n, int nrhs, float* ap, int* ipiv, float* b, int ldb);

        /// <summary>
        /// Computes the solution to the system of linear equations 
        /// with a real or complex symmetric single precision matrix A stored in packed format,
        /// and multiple right-hand sides.
        /// </summary>
        /// <param name="uplo">Must be 'U' or 'L'. Indicates whether the upper or lower triangular part of A is stored:
        /// If uplo = 'U', the upper triangle of A is stored.
        /// If uplo = 'L', the lower triangle of A is stored.</param>
        /// <param name="n">The order of matrix A; n ≥ 0.</param>
        /// <param name="nrhs">The number of right-hand sides, the number of columns in B; nrhs ≥ 0.</param>
        /// <param name="a">ap, b: Arrays: ap (size max(1,n*(n+1)/2), b of size max(1, ldb*nrhs) for column major layout and max(1, ldb*n) for row major layout.
        /// The array ap contains the factor U or L, as specified by uplo, in packed storage(see Matrix Storage Schemes).
        /// The array b contains the matrix B whose columns are the right-hand sides for the systems of equations.</param>
        /// <param name="ipiv">Out: Array, size at least max(1, n). Contains details of the interchanges and the block structure of D, as determined by ?sptrf.
        /// If ipiv[i - 1] = k > 0, then dii is a 1-by-1 block, and the i-th row and column of A was interchanged with the k-th row and column.
        /// If uplo = 'U' and ipiv[i]= ipiv[i - 1] = -m &lt; 0, then D has a 2-by-2 block in rows/columns i and i+1, and i-th row and column of A was interchanged with the m-th row and column.
        /// If uplo = 'L' and ipiv[i-1] = ipiv[i] = -m &lt; 0, then D has a 2-by-2 block in rows/columns i and i+1, and (i+1)-th row and column of A was interchanged with the m-th row and column.</param>
        /// <param name="b">see ap</param>
        /// <param name="ldb">The leading dimension of b; ldb ≥ max(1, n) for column major layout and ldb ≥ nrhs for row major layout.</param>
        public void sspsv(char uplo, int n, int nrhs, float[] a, int[] ipiv,
            float[] b, int ldb)
        {
            fixed (float* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            fixed (float* pB = &b[0])
            {
                CheckInfo(LAPACKE_sspsv(LAPACK_ROW_MAJOR, uplo, n, nrhs, pA, pIpiv, pB, ldb));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dspsv(int matrix_layout, char uplo, int n, int nrhs,
            double* ap, int* ipiv, double* b, int ldb);

        /// <summary>
        /// Computes the solution to the system of linear equations 
        /// with a real or complex symmetric double precision matrix A stored in packed format,
        /// and multiple right-hand sides.
        /// </summary>
        /// <param name="uplo">Must be 'U' or 'L'. Indicates whether the upper or lower triangular part of A is stored:
        /// If uplo = 'U', the upper triangle of A is stored.
        /// If uplo = 'L', the lower triangle of A is stored.</param>
        /// <param name="n">The order of matrix A; n ≥ 0.</param>
        /// <param name="nrhs">The number of right-hand sides, the number of columns in B; nrhs ≥ 0.</param>
        /// <param name="a">ap, b: Arrays: ap (size max(1,n*(n+1)/2), b of size max(1, ldb*nrhs) for column major layout and max(1, ldb*n) for row major layout.
        /// The array ap contains the factor U or L, as specified by uplo, in packed storage(see Matrix Storage Schemes).
        /// The array b contains the matrix B whose columns are the right-hand sides for the systems of equations.</param>
        /// <param name="ipiv">Out: Array, size at least max(1, n). Contains details of the interchanges and the block structure of D, as determined by ?sptrf.
        /// If ipiv[i - 1] = k > 0, then dii is a 1-by-1 block, and the i-th row and column of A was interchanged with the k-th row and column.
        /// If uplo = 'U' and ipiv[i]= ipiv[i - 1] = -m &lt; 0, then D has a 2-by-2 block in rows/columns i and i+1, and i-th row and column of A was interchanged with the m-th row and column.
        /// If uplo = 'L' and ipiv[i-1] = ipiv[i] = -m &lt; 0, then D has a 2-by-2 block in rows/columns i and i+1, and (i+1)-th row and column of A was interchanged with the m-th row and column.</param>
        /// <param name="b">see ap</param>
        /// <param name="ldb">The leading dimension of b; ldb ≥ max(1, n) for column major layout and ldb ≥ nrhs for row major layout.</param>
        public void dspsv(char uplo, int n, int nrhs, double[] a, int[] ipiv,
            double[] b, int ldb)
        {
            fixed (double* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            fixed (double* pB = &b[0])
            {
                CheckInfo(LAPACKE_dspsv(LAPACK_ROW_MAJOR, uplo, n, nrhs, pA, pIpiv, pB, ldb));
            }
        }


        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dtrtrs(int matrix_order, char uplo, char trans,
            char diag, int n, int nrhs, double* a, int lda, double* b, int ldb);

        /// <summary>
        /// solves a system of linear equation with triangular double precision matrix
        /// find X s.t. A.X = B
        /// </summary>
        /// <param name="uplo"> 'U' if A is upper triangular, 'L' if lower triangular </param>
        /// <param name="trans">
        /// 'N' to solve A . X = B
        /// 'T' to solve A^T . X = B
        /// 'C' to solve A^H . X = B
        /// </param>
        /// <param name="diag">
        /// 'N' if A is not unit triangular
        /// 'U' if diagonal elt of A are 1s
        /// </param>
        /// <param name="n"> number of rows in a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="b"> right hand side of the equation </param>
        /// <param name="ldb"> leading dimension of b </param>
        public void dtrtrs(char uplo, char trans, char diag, int n, int nrhs,
            double[] a, int lda, double[] b, int ldb)
        {
            fixed (double* pA = &a[0])
            fixed (double* pB = &b[0])
            {
                CheckInfo(LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, uplo, trans, diag, n, nrhs, pA, lda, pB, ldb));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_strtrs(int matrix_order, char uplo, char trans,
            char diag, int n, int nrhs, float* a, int lda, float* b, int ldb);

        /// <summary>
        /// solves a system of linear equation with triangular single precision matrix
        /// find X s.t. A.X = B
        /// </summary>
        /// <param name="uplo"> 'U' if A is upper triangular, 'L' if lower triangular </param>
        /// <param name="trans">
        /// 'N' to solve A . X = B
        /// 'T' to solve A^T . X = B
        /// 'C' to solve A^H . X = B
        /// </param>
        /// <param name="diag">
        /// 'N' if A is not unit triangular
        /// 'U' if diagonal elt of A are 1s
        /// </param>
        /// <param name="n"> number of rows in a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="b"> right hand side of the equation </param>
        /// <param name="ldb"> leading dimension of b </param>
        public void strtrs(char uplo, char trans, char diag, int n, int nrhs,
            float[] a, int lda, float[] b, int ldb)
        {
            fixed (float* pA = &a[0])
            fixed (float* pB = &b[0])
            {
                CheckInfo(LAPACKE_strtrs(LAPACK_ROW_MAJOR, uplo, trans, diag, n, nrhs, pA, lda, pB, ldb));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgesv(int matrix_order, int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb);

        /// <summary>
        /// solve a system of linear equation with general double precision square matrix
        /// </summary>
        /// <param name="n"> number of rows in a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="ipiv"> temporary storage </param> 
        /// <param name="b"> right hand side of the equation </param>
        /// <param name="ldb"> leading dimension of b </param>
        public void dgesv(int n, int nrhs, double[] a, int lda, int[] ipiv, double[] b, int ldb)
        {
            fixed (double* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            fixed (double* pB = &b[0])
            {
                CheckInfo(LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, pA, lda, pIpiv, pB, ldb));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgesv(int matrix_order, int n, int nrhs, float* a,
            int lda, int* ipiv, float* b, int ldb);

        /// <summary>
        /// solve a system of linear equation with general single precision square matrix
        /// </summary>
        /// <param name="n"> number of rows in a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="ipiv"> temporary storage </param> 
        /// <param name="b"> right hand side of the equation </param>
        /// <param name="ldb"> leading dimension of b </param>
        public void sgesv(int n, int nrhs, float[] a, int lda, int[] ipiv, float[] b, int ldb)
        {
            fixed (float* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            fixed (float* pB = &b[0])
            {
                CheckInfo(LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, pA, lda, pIpiv, pB, ldb));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgelss(int matrix_layout, int m, int n, int nrhs,
            double* a, int lda, double* b, int ldb, double* s, float rcond, int* rank);

        /// <summary>
        /// Compute the minimum norm solution to a double precision linear least squares problem
        /// Uses SVD decomposition
        /// </summary>
        /// <param name="m"> number of rows in a </param>
        /// <param name="n"> number of columns in a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="b"> right hand side matrix b </param>
        /// <param name="ldb"> leading dimension of b </param>
        /// <param name="s"> temporary storage </param>
        /// <param name="rcond"> used to dtermine the effective rank of a </param>
        /// <param name="rank"> temporary storage </param>
        public void dgelss(int m, int n, int nrhs,
            double[] a, int lda, double[] b, int ldb,
            double[] s, float rcond, ref int rank)
        {
            fixed (double* pA = &a[0])
            fixed (double* pB = &b[0])
            fixed (double* pS = &s[0])
            fixed (int* pRank = &rank)
            {
                CheckInfo(LAPACKE_dgelss(LAPACK_ROW_MAJOR, m, n, nrhs, pA, lda, pB, ldb, pS, rcond, pRank));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgelss(int matrix_layout, int m, int n, int nrhs,
            float* a, int lda, float* b, int ldb,
            float* s, float rcond, int* rank);

        /// <summary>
        /// Compute the minimum norm solution to a single precision linear least squares problem
        /// Uses SVD decomposition
        /// </summary>
        /// <param name="m"> number of rows in a </param>
        /// <param name="n"> number of columns in a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="b"> right hand side matrix b </param>
        /// <param name="ldb"> leading dimension of b </param>
        /// <param name="s"> temporary storage </param>
        /// <param name="rcond"> used to dtermine the effective rank of a </param>
        /// <param name="rank"> temporary storage </param>
        public void sgelss(int m, int n, int nrhs,
            float[] a, int lda, float[] b, int ldb,
            float[] s, float rcond, ref int rank)
        {
            fixed (float* pA = &a[0])
            fixed (float* pB = &b[0])
            fixed (float* pS = &s[0])
            fixed (int* pRank = &rank)
            {
                CheckInfo(LAPACKE_sgelss(LAPACK_ROW_MAJOR, m, n, nrhs, pA, lda, pB, ldb, pS, rcond, pRank));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgels(int matrix_layout, char trans, int m, int n,
            int nrhs, double* a, int lda, double* b, int ldb);

        /// <summary>
        ///  Uses QR or LQ factorization to solve an overdetermined or underdetermined linear
        ///  system with full rank double precision matrix a.
        /// </summary>
        /// <param name="trans">
        /// 'N'if the linear system involve A,
        /// 'T' if it involve A^T
        /// 'C' if A is complex and the system involves A^H
        /// </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="b"> right hand side of the equation </param>
        /// <param name="ldb"> leading dimension of b </param>
        public void dgels(char trans, int m, int n, int nrhs, double[] a, int lda, double[] b, int ldb)
        {
            fixed (double* pA = &a[0])
            fixed (double* pB = &b[0])
            {
                CheckInfo(LAPACKE_dgels(LAPACK_ROW_MAJOR, trans, m, n, nrhs, pA, lda, pB, ldb));
            }
        }


        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgels(int matrix_layout, char trans, int m, int n,
            int nrhs, float* a, int lda, float* b, int ldb);

        /// <summary>
        ///  Uses QR or LQ factorization to solve an overdetermined or underdetermined linear
        ///  system with full rank single precision matrix a.
        /// </summary>
        /// <param name="trans">
        /// 'N'if the linear system involve A,
        /// 'T' if it involve A^T
        /// 'C' if A is complex and the system involves A^H
        /// </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="nrhs"> number of right hand side (columns of b) </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="b"> right hand side of the equation </param>
        /// <param name="ldb"> leading dimension of b </param>
        public void sgels(char trans, int m, int n, int nrhs,
            float[] a, int lda, float[] b, int ldb)
        {
            fixed (float* pA = &a[0])
            fixed (float* pB = &b[0])
            {
                CheckInfo(LAPACKE_sgels(LAPACK_ROW_MAJOR, trans, m, n, nrhs, pA, lda, pB, ldb));
            }
        }

        /////////////////////////////////////
        //// Singular value decomposition ///
        /////////////////////////////////////

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgesvd(int matrix_layout, char jobu, char jobvt,
                    int m, int n, float* a, int lda, float* s, float* u, int ldu, float* vt, int ldvt,
                    float* superb);

        /// <summary>
        /// computes the singular value decomposition (SVD) 
        /// of a general rectangular single precision m * n matrix A,
        /// optionally the left and/or right singular vectors.
        /// A = U * Sigma * V^T 
        /// </summary>
        /// <param name="jobu">
        /// 'A' (all m col of U are returned in u), 
        /// 'S' (min(m,n) col of U are returned in u), 
        /// 'O' (min(m,n) col of U overwritten in a),
        /// or 'N' (no col of U are computed)
        /// </param>
        /// <param name="jobvt">
        /// 'A' (all n col of V^T are returned in vt), 
        /// 'S' (min(m,n) col of V^T are returned in vt), 
        /// 'O' (min(m,n) col of V^T overwritten in a),
        /// or 'N' (no col of V^T are computed)
        /// </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="a"> matrix a</param>
        /// <param name="lda"> leading dim of a </param>
        /// <param name="s"> sigma diagonal matrix filled during calculations </param>
        /// <param name="u"> left matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U</param>
        /// <param name="vt"> right matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V </param>
        /// <param name="superb"> temporary storage </param>
        public void sgesvd(char jobu, char jobvt,
            int m, int n, float[] a,
            int lda, float[] s, float[] u, int ldu,
            float[] vt, int ldvt, float[] superb)
        {
            fixed (float* pA = &a[0])
            fixed (float* pS = &s[0])
            fixed (float* pU = &u[0])
            fixed (float* pVt = &vt[0])
            fixed (float* pSuperb = &superb[0])
            {
                CheckInfo(LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, pA, lda,
                    pS, pU, ldu, pVt, ldvt, pSuperb));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgesvd(int matrix_layout, char jobu, char jobvt, int m,
            int n, double* a, int lda, double* s, double* u, int ldu,
            double* vt, int ldvt, double* superb);

        /// <summary>
        /// computes the singular value decomposition (SVD) 
        /// of a general rectangular double precision m * n matrix A,
        /// optionally the left and/or right singular vectors.
        /// A = U * Sigma * V^T 
        /// </summary>
        /// <param name="jobu">
        /// 'A' (all m col of U are returned in u), 
        /// 'S' (min(m,n) col of U are returned in u), 
        /// 'O' (min(m,n) col of U overwritten in a),
        /// or 'N' (no col of U are computed)
        /// </param>
        /// <param name="jobvt">
        /// 'A' (all n col of V^T are returned in vt), 
        /// 'S' (min(m,n) col of V^T are returned in vt), 
        /// 'O' (min(m,n) col of V^T overwritten in a),
        /// or 'N' (no col of V^T are computed)
        /// </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="a"> matrix a</param>
        /// <param name="lda"> leading dim of a </param>
        /// <param name="s"> sigma diagonal matrix filled during calculations </param>
        /// <param name="u"> left matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U</param>
        /// <param name="vt"> right matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V </param>
        /// <param name="superb"> temporary storage </param>
        public void dgesvd(char jobu, char jobvt, int m, int n, double[] a, int lda,
            double[] s, double[] u, int ldu, double[] vt, int ldvt, double[] superb)
        {
            fixed (double* pA = &a[0])
            fixed (double* pS = &s[0])
            fixed (double* pU = &u[0])
            fixed (double* pVt = &vt[0])
            fixed (double* pSuperb = &superb[0])
            {
                CheckInfo(LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, m, n, pA, lda, pS, pU, ldu, pVt, ldvt, pSuperb));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgesdd(int matrix_order, char jobz, int m,
            int n, double* a, int lda, double* s,
            double* u, int ldu, double* vt,
            int ldvt);

        /// <summary>
        /// Singular Value Decomposition for general double precision m * n rectangular matrix
        /// A = U*D*V^T
        /// </summary>
        /// <param name="jobz">
        /// 'A' all m columns of U and all n rows of V^T are returned in u and vt
        /// 'S' first min(m,n) columns of U and min(m,n) rows of V^T are returned in u and vt
        /// 'O' if m >= n first n columns of U overwritten in a, all rows of V^T written in vt
        ///     if m &lt; n all columns of U returned in u, first m rows of V^T overwritten in a
        /// 'N' no columns of U and rows of V^T computed
        /// </param>
        /// <param name="m"> number of rows in a </param>
        /// <param name="n"> number of columns in a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="s"> sigma diagonal matrix of SVD filled during calculations </param>
        /// <param name="u"> U matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U </param>
        /// <param name="vt"> V^T matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V^T </param>
        public void dgesdd(char jobz, int m, int n, double[] a, int lda, double[] s,
            double[] u, int ldu, double[] vt, int ldvt)
        {
            fixed (double* pA = &a[0])
            fixed (double* pS = &s[0])
            fixed (double* pU = &u[0])
            fixed (double* pVt = &vt[0])
            {
                CheckInfo(LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, pA, lda, pS, pU, ldu, pVt, ldvt));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgesdd(int matrix_order, char jobz, int m,
            int n, float* a, int lda, float* s,
            float* u, int ldu, float* vt,
            int ldvt);

        /// <summary>
        /// Singular Value Decomposition for general single precision m * n rectangular matrix
        /// A = U*D*V^T
        /// </summary>
        /// <param name="jobz">
        /// 'A' all m columns of U and all n rows of V^T are returned in u and vt
        /// 'S' first min(m,n) columns of U and min(m,n) rows of V^T are returned in u and vt
        /// 'O' if m >= n first n columns of U overwritten in a, all rows of V^T written in vt
        ///     if m &lt; n all columns of U returned in u, first m rows of V^T overwritten in a
        /// 'N' no columns of U and rows of V^T computed
        /// </param>
        /// <param name="m"> number of rows in a </param>
        /// <param name="n"> number of columns in a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="s"> sigma diagonal matrix of SVD filled during calculations </param>
        /// <param name="u"> U matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U </param>
        /// <param name="vt"> V^T matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V^T </param>
        public void sgesdd(char jobz, int m, int n, float[] a, int lda, float[] s,
            float[] u, int ldu, float[] vt, int ldvt)
        {
            fixed (float* pA = &a[0])
            fixed (float* pS = &s[0])
            fixed (float* pU = &u[0])
            fixed (float* pVt = &vt[0])
            {
                CheckInfo(LAPACKE_sgesdd(LAPACK_ROW_MAJOR, jobz, m, n, pA, lda, pS, pU, ldu, pVt, ldvt));
            }
        }

        ////////////////////////////////////
        ////   Matrix Factorization   //////
        ////////////////////////////////////

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgetrf(int matrix_layout, int m, int n, float* a, int lda, int* ipiv);

        /// <summary>
        /// The routine computes the LU factorization 
        /// of a general m-by-n single precision matrix A as
        /// A = P*L*U,
        /// where P is a permutation matrix, L is lower triangular with unit diagonal elements(lower trapezoidal if m > n)
        /// and U is upper triangular(upper trapezoidal if m &lt; n). The routine uses partial pivoting, with row interchanges.
        /// </summary>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="ipiv"> necessary to store intermediate results </param>
        public void sgetrf(int m, int n, float[] a, int lda, int[] ipiv)
        {
            fixed (float* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, pA, lda, pIpiv));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgetrf(int matrix_layout, int m, int n, double* a,
            int lda, int* ipiv);

        /// <summary>
        /// The routine computes the LU factorization of a general m-by-n double precision matrix A as
        /// A = P*L*U,
        /// where P is a permutation matrix, L is lower triangular with unit diagonal elements(lower trapezoidal if m > n)
        /// and U is upper triangular(upper trapezoidal if m &lt; n). The routine uses partial pivoting, with row interchanges.
        /// </summary>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="ipiv"> temporary storage </param>
        public void dgetrf(int m, int n, double[] a, int lda, int[] ipiv)
        {
            fixed (double* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, pA, lda, pIpiv));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgetri(int matrix_layout, int n, float* a,
            int lda, int* ipiv);

        /// <summary>
        /// Computes the inverse of an LU-factored general single precision matrix.
        /// The routine computes the inverse inv(A) of a general matrix A. Before calling this routine, call sgetrf to factorize A.
        /// </summary>
        /// <param name="n">The order of the matrix A; n ≥ 0.</param>
        /// <param name="a">Array a(size max(1, lda* n)) contains the factorization of the matrix A, as returned by sgetrf: A = P*L*U. 
        /// The second dimension of a must be at least max(1, n).
        /// Overwritten by the n-by-n matrix inv(A).
        /// </param>
        /// <param name="lda">The leading dimension of a; lda ≥ max(1, n).</param>
        /// <param name="ipiv">Array, size at least max(1, n). The ipiv array, as returned by ?getrf.</param>
        public void sgetri(int n, float[] a, int lda, int[] ipiv)
        {
            fixed (float* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, pA, lda, pIpiv));
            }
        }


        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgetri(int matrix_layout, int n, double* a, int lda, int* ipiv);

        /// <summary>
        /// Computes the inverse of an LU-factored general double precision matrix.
        /// The routine computes the inverse inv(A) of a general matrix A. Before calling this routine, call dgetrf to factorize A.
        /// </summary>
        /// <param name="n">The order of the matrix A; n ≥ 0.</param>
        /// <param name="a">Array a(size max(1, lda* n)) contains the factorization of the matrix A, as returned by dgetrf: A = P*L*U. 
        /// The second dimension of a must be at least max(1, n).
        /// Overwritten by the n-by-n matrix inv(A).
        /// </param>
        /// <param name="lda">The leading dimension of a; lda ≥ max(1, n).</param>
        /// <param name="ipiv">Array, size at least max(1, n). The ipiv array, as returned by ?getrf.</param>
        public void dgetri(int n, double[] a, int lda, int[] ipiv)
        {
            fixed (double* pA = &a[0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, pA, lda, pIpiv));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dpotrf(int matrix_order, char uplo, int n,
            double* a, int lda);

        /// <summary>
        /// Cholesky decomposition (double precision array)
        /// </summary>
        /// <param name="uplo"> 'U' if upper triangle is stored in a, else 'L' </param>
        /// <param name="n"> second dimension of a </param>
        /// <param name="a"> symetric matrix </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="offset"> starting offset in a </param>
        public void dpotrf(char uplo, int n, double[] a, int lda, int offset = 0)
        {
            fixed (double* pA = &a[offset])
            {
                CheckInfo(LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, pA, lda));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_spotrf(int matrix_order, char uplo, int n, float* a, int lda);

        /// <summary>
        /// Cholesky decomposition (single precision array)
        /// </summary>
        /// <param name="uplo"> 'U' if upper triangle is stored in a, else 'L' </param>
        /// <param name="n"> second dimension of a </param>
        /// <param name="a"> symetric matrix </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="offset"> starting offset in a </param>
        /// <returns></returns>
        public void spotrf(char uplo, int n, float[] a, int lda, int offset = 0)
        {
            fixed (float* pA = &a[offset])
            {
                CheckInfo(LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, n, pA, lda));
            }
        }

        ////////////////////////////////////
        ////  Eigen Decomposition      /////
        ////////////////////////////////////

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dsyev(int matrix_layout, char jobz, char uplo, int n,
                    double* a, int lda, double* w);

        /// <summary>
        /// Compute all eigenvalues and optionnaly eigenvectors
        /// of a double precision matrix a
        /// </summary>
        /// <param name="jobz">
        /// 'N' only eigenvalues are computed
        /// 'V' eigenvalues and eigenvectors are computed 
        /// </param>
        /// <param name="uplo">'U' if upper triangle is stored in a, 'L' if lower triangle is stored </param>
        /// <param name="n"> size of a </param>
        /// <param name="a"> symetric matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="w"> list of eigenvectors filled during process </param>
        public void dsyev(char jobz, char uplo, int n, double[] a, int lda, double[] w)
        {
            fixed (double* pA = &a[0])
            fixed (double* pW = &w[0])
            {
                CheckInfo(LAPACKE_dsyev(LAPACK_ROW_MAJOR, jobz, uplo, n, pA, lda, pW));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_ssyev(int matrix_layout, char jobz, char uplo, int n,
            float* a, int lda, float* w);

        /// <summary>
        /// Compute all eigenvalues and optionnaly eigenvectors
        /// of a single precision matrix a
        /// </summary>
        /// <param name="jobz">
        /// 'N' only eigenvalues are computed
        /// 'V' eigenvalues and eigenvectors are computed 
        /// </param>
        /// <param name="uplo">'U' if upper triangle is stored in a, 'L' if lower triangle is stored </param>
        /// <param name="n"> size of a </param>
        /// <param name="a"> symetric matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="w"> list of eigenvectors filled during process </param>
        public void ssyev(char jobz, char uplo, int n, float[] a, int lda, float[] w)
        {
            fixed (float* pA = &a[0])
            fixed (float* pW = &w[0])
            {
                CheckInfo(LAPACKE_ssyev(LAPACK_ROW_MAJOR, jobz, uplo, n, pA, lda, pW));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dsyevr(int matrix_layout, int jobz, char range, char uplo,
            int n, double* a, int lda, double vl, double vu, int il, int iu, float abstol,
            ref int m, double* w, double* z, int ldz, int* isuppz);

        /// <summary>
        /// Compute selected eigenvalues and optionnally eigenvectors 
        /// of a double precision symetric matrix a
        /// </summary>
        /// <param name="jobz"> 'N' to compute only eigenvalues, 'V' to add eigenvectors </param>
        /// <param name="range"> 
        /// 'A' for all eigenvalues,
        /// 'I' for eigenvalues with indices between il and iu,
        /// 'V' for eigenvalues whose value is between vl and vu
        /// </param>
        /// <param name="uplo"> 'U' if upper triangle of a is stored, 'L' if lower triangle </param>
        /// <param name="n"> order of a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="vl"> min value for eigenvalue (if range = 'V') </param>
        /// <param name="vu"> max value for eigenvalue (if range = 'V') </param>
        /// <param name="il"> min index for eigenvalue (if range = 'I') </param>
        /// <param name="iu"> max index for eigenvalue (if range = 'I') </param>
        /// <param name="abstol"> control bound if jobz = 'V' </param>
        /// <param name="m"> number of eigenvalues returned </param>
        /// <param name="w"> contains the eigenvalues after thr process </param>
        /// <param name="z"> contains the eigenvectors if jobz = 'V' </param>
        /// <param name="ldz"> leading dimension of z </param>
        /// <param name="isuppz"> temporary storage used when computing eigenvectors </param>
        public void dsyevr(int jobz, char range, char uplo, int n, double[] a, int lda,
            double vl, double vu, int il, int iu, float abstol, ref int m, double[] w,
            double[] z, int ldz, int[] isuppz)
        {
            fixed (double* pA = &a[0])
            fixed (double* pZ = &z[0])
            fixed (double* pW = &w[0])
            fixed (int* pIsuppz = &isuppz[0])
            {
                CheckInfo(LAPACKE_dsyevr(LAPACK_ROW_MAJOR, jobz, range, uplo, n, pA, lda, vl, vu,
                    il, iu, abstol, ref m, pW, pZ, ldz, pIsuppz));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_ssyevr(int matrix_layout, int jobz, char range, char uplo,
            int n, float* a, int lda, float vl, float vu, int il, int iu, float abstol,
            ref int m, float* w, float* z, int ldz, int* isuppz);

        /// <summary>
        /// Compute selected eigenvalues and optionnally eigenvectors 
        /// of a single precision symetric matrix a
        /// </summary>
        /// <param name="jobz"> 'N' to compute only eigenvalues, 'V' to add eigenvectors </param>
        /// <param name="range"> 
        /// 'A' for all eigenvalues,
        /// 'I' for eigenvalues with indices between il and iu,
        /// 'V' for eigenvalues whose value is between vl and vu
        /// </param>
        /// <param name="uplo"> 'U' if upper triangle of a is stored, 'L' if lower triangle </param>
        /// <param name="n"> order of a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="vl"> min value for eigenvalue (if range = 'V') </param>
        /// <param name="vu"> max value for eigenvalue (if range = 'V') </param>
        /// <param name="il"> min index for eigenvalue (if range = 'I') </param>
        /// <param name="iu"> max index for eigenvalue (if range = 'I') </param>
        /// <param name="abstol"> control bound if jobz = 'V' </param>
        /// <param name="m"> number of eigenvalues returned </param>
        /// <param name="w"> contains the eigenvalues after thr process </param>
        /// <param name="z"> contains the eigenvectors if jobz = 'V' </param>
        /// <param name="ldz"> leading dimension of z </param>
        /// <param name="isuppz"> temporary storage used when computing eigenvectors </param>
        public void ssyevr(int jobz, char range, char uplo, int n, float[] a, int lda,
            float vl, float vu, int il, int iu, float abstol, ref int m, float[] w,
            float[] z, int ldz, int[] isuppz)
        {
            fixed (float* pA = &a[0])
            fixed (float* pZ = &z[0])
            fixed (float* pW = &w[0])
            fixed (int* pIsuppz = &isuppz[0])
            {
                CheckInfo(LAPACKE_ssyevr(LAPACK_ROW_MAJOR, jobz, range, uplo, n, pA, lda, vl, vu,
                    il, iu, abstol, ref m, pW, pZ, ldz, pIsuppz));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_dgeev(int matrix_layout, char jobvl, char jobvr, int n,
               double* a, int lda, double* wr, double* wi, double* vl,
               int ldvl, double* vr, int ldvr);

        /// <summary>
        /// Compute eigenvalues and optionnally eigenvectors
        /// of a general double precision square matrix
        /// </summary>
        /// <param name="jobvl"> 'N': left eigenvectors are not computed, 'V' they are </param>
        /// <param name="jobvr"> 'N': right eigenvectors are not computed, 'V' they are </param>
        /// <param name="n"> order of matrix a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="wr"> store real parts of eigenvalues </param>
        /// <param name="wi"> store imaginary part of eigenvalues </param>
        /// <param name="vl"> store the left eigenvectors if jobvl = 'V' </param>
        /// <param name="ldvl"> leading dimension of vl </param>
        /// <param name="vr"> store the right eigenvectors if jobvr = 'V' </param>
        /// <param name="ldvr"> leading dimension of vr </param>
        public void dgeev(char jobvl, char jobvr, int n, double[] a, int lda, double[] wr,
            double[] wi, double[] vl, int ldvl, double[] vr, int ldvr)
        {
            fixed (double* pA = &a[0])
            fixed (double* pWr = &wr[0])
            fixed (double* pWi = &wi[0])
            fixed (double* pVl = &vl[0])
            fixed (double* pVr = &vr[0])
            {
                CheckInfo(LAPACKE_dgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, pA, lda, pWr, pWi,
                    pVl, ldvl, pVr, ldvr));
            }
        }

        [DllImport(MKLRT_WINDOWS_DLL, CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        internal static extern int LAPACKE_sgeev(int matrix_layout, char jobvl, char jobvr, int n,
            float* a, int lda, float* wr, float* wi, float* vl,
            int ldvl, float* vr, int ldvr);

        /// <summary>
        /// Compute eigenvalues and optionnally eigenvectors
        /// of a general single precision square matrix
        /// </summary>
        /// <param name="jobvl"> 'N': left eigenvectors are not computed, 'V' they are </param>
        /// <param name="jobvr"> 'N': right eigenvectors are not computed, 'V' they are </param>
        /// <param name="n"> order of matrix a </param>
        /// <param name="a"> matrix a </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="wr"> store real parts of eigenvalues </param>
        /// <param name="wi"> store imaginary part of eigenvalues </param>
        /// <param name="vl"> store the left eigenvectors if jobvl = 'V' </param>
        /// <param name="ldvl"> leading dimension of vl </param>
        /// <param name="vr"> store the right eigenvectors if jobvr = 'V' </param>
        /// <param name="ldvr"> leading dimension of vr </param>
        public void sgeev(char jobvl, char jobvr, int n, float[] a, int lda, float[] wr,
            float[] wi, float[] vl, int ldvl, float[] vr, int ldvr)
        {
            fixed (float* pA = &a[0])
            fixed (float* pWr = &wr[0])
            fixed (float* pWi = &wi[0])
            fixed (float* pVl = &vl[0])
            fixed (float* pVr = &vr[0])
            {
                CheckInfo(LAPACKE_sgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, pA, lda, pWr, pWi,
                    pVl, ldvl, pVr, ldvr));
            }
        }

        /// <summary>
        /// Check if routine went well
        /// </summary>
        /// <param name="info"></param>
        public static void CheckInfo(int info)
        {
            if (info > 0)
                // If info = i, Dii is 0
                // The factorization has been completed, but D is exactly singular.
                // Division by 0 will occur if you use D for solving a system of linear equations.
                throw new ArgumentException($"Matrix is singular: {info}-th leading minor not positive definite");
            if (info < 0)
                throw new ArgumentException($"Invalid parameter #{-info}");
        }

        /////////////////////////////////////////////////
        ////   Higher level functions ready to use   ////
        /////////////////////////////////////////////////


        /// <summary>
        /// Returns the determinant of a given general single precision square matrix of order n
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <returns></returns>
        public float Determinant(float[] a, int n)
        {
            var ipiv = new int[n];
            sgetrf(n, n, a, n, ipiv);
            var det = 1f;
            for (int i = 0; i < n; i++)
            {
                det *= a[i * n + i];
                if (ipiv[i] - 1 != i) det = -det;
            }
            return det;
        }

        /// <summary>
        /// Returns the determinant of a given general double precision square matrix of order n
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <returns></returns>
        public double Determinant(double[] a, int n)
        {
            var ipiv = new int[n];
            dgetrf(n, n, a, n, ipiv);
            var det = 1.0;
            for (int i = 0; i < n; i++)
            {
                det *= a[i * n + i];
                if (ipiv[i] - 1 != i) det = -det;
            }
            return det;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="n"></param>
        /// <param name="offset"></param>
        /// <param name="lda"></param>
        /// <returns></returns>
        public Tuple<float, float> SLogDet(float[] a, int n, int offset = 0, int lda = 0)
        {
            // https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/309460

            if (lda == 0) lda = n;

            var ipiv = new int[n];

            fixed (float* pA = &a[offset])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, pA, lda, pIpiv));
            }
            var s = 1f;
            var det = 0.0;
            for (int i = 0; i < n; i++)
            {
                det += Math.Log(a[i * n + i]);
                if (ipiv[i] - 1 != i) s = -s;
            }

            return Tuple.Create(s, (float)det);
        }

        /// <summary>
        /// return the determinant of a single precision square matrix
        /// represented as a 2D C# array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> the ordrer of a </param>
        /// <returns> determinant of a </returns>
        public float Determinant(float[,] a, int n)
        {
            // https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/309460
            var ipiv = new int[n];
            fixed (float* pA = &a[0, 0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, pA, n, pIpiv));
            }
            var det = 1f;
            for (int i = 0; i < n; i++)
            {
                det *= a[i, i];
                if (ipiv[i] - 1 != i) det = -det;
            }
            return det;
        }

        /// <summary>
        /// return the determinant of a double precision matrix
        /// represented as a 2D C# array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> the ordrer of a </param>
        /// <returns> determinant of a </returns>
        public double Determinant(double[,] a, int n)
        {
            // https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/309460
            var ipiv = new int[n];
            fixed (double* pA = &a[0, 0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, pA, n, pIpiv));
            }
            var det = 1.0;
            for (int i = 0; i < n; i++)
            {
                det *= a[i, i];
                if (ipiv[i] - 1 != i) det = -det;
            }
            return det;
        }

        /// <summary>
        /// Inverse inplace a given single precision matrix represented as a 2D array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        public void Inverse(float[,] a, int n)
        {
            var ipiv = new int[n + 1];
            fixed (float* pA = &a[0, 0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, pA, n, pIpiv));    // single general matrix, see ssytrf for single symetric matrix
                CheckInfo(LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, pA, n, pIpiv));
            }
        }

        /// <summary>
        /// Inverse inplace a given double precision matrix represented as a 2D array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        public void Inverse(double[,] a, int n)
        {
            var ipiv = new int[n + 1];
            fixed (double* pA = &a[0, 0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, pA, n, pIpiv));    // single general matrix, see ssytrf for single symetric matrix
                CheckInfo(LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, pA, n, pIpiv));
            }
        }

        /// <summary>
        /// Inverse inplace a single precision matrix 
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <param name="offset"> starting offset of a </param>
        /// <param name="lda"> leading dimension of a</param>
        public void Inverse(float[] a, int n, int offset = 0, int lda = 0)
        {
            if (lda == 0) lda = n;
            var ipiv = new int[n + 1];
            fixed (float* pA = &a[offset])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, pA, lda, pIpiv));    // single general matrix, see ssytrf for single symetric matrix
                CheckInfo(LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, pA, lda, pIpiv));
            }
        }

        /// <summary>
        /// Inverse inplace a double precision matrix 
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <param name="offset"> starting offset of a </param>
        /// <param name="lda"> leading dimension of a</param>
        public void Inverse(double[] a, int n, int offset = 0, int lda = 0)
        {
            if (lda == 0) lda = n;
            var ipiv = new int[n + 1];
            fixed (double* pA = &a[offset])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, pA, lda, pIpiv));    // single general matrix, see ssytrf for single symetric matrix
                CheckInfo(LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, pA, lda, pIpiv));
            }
        }

        /// <summary>
        /// Inverse single precision matrix inplace
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        public void Inverse(double[] a, int n)
        {
            var ipiv = new int[n + 1];
            dgetrf(n, n, a, n, ipiv);
            dgetri(n, a, n, ipiv);
        }

        /// <summary>
        /// Inverse double precision matrix inplace
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        public void Inverse(float[] a, int n)
        {
            var ipiv = new int[n + 1];
            sgetrf(n, n, a, n, ipiv);
            sgetri(n, a, n, ipiv);
        }

        /// <summary>
        /// Inverse double matrix precision inplace
        /// </summary>
        /// <param name="a"></param>
        /// <param name="n"></param>
        /// <param name="m"></param>
        public void Inverse(double[,] a, int n, int m)
        {
            var ipiv = new int[n + 1];
            fixed (double* pA = &a[0, 0])
            fixed (int* pIpiv = &ipiv[0])
            {
                CheckInfo(LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, m, pA, n, pIpiv));    // double general matrix, see dsytrf for double symetric matrix
                CheckInfo(LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, pA, n, pIpiv));
            }
        }

        /// <summary>
        /// Return the list of eigenvalues of a double precision symetric matrix.
        /// Matrix should be store in row major fashion and only upper triangle need be stored
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <returns></returns>
        public double[] EigenValuesSymetric(double[] a, int n)
        {
            var w = new double[n];
            char jobz = 'N';
            char uplo = 'U';
            dsyev(jobz, uplo, n, a, n, w);
            return w;
        }

        /// <summary>
        /// Return the list of eigenvalues of a single precision symetric matrix.
        /// Matrix should be store in row major fashion and only upper triangle need be stored
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <returns></returns>
        public float[] EigenValuesSymetric(float[] a, int n)
        {
            var w = new float[n];
            char jobz = 'N';
            char uplo = 'U';
            ssyev(jobz, uplo, n, a, n, w);
            return w;
        }

        /// <summary>
        /// Compute and return the kth first eigenvectors of a double precision symetric matrix.
        /// Matrix A should be stored in a row major fashion.
        /// </summary>
        /// <param name="a"> symetric matrix </param>
        /// <param name="n"> leading dimension of A </param>
        /// <param name="k"> number of eigen vectors to compute (between 1 and n) </param>
        /// <param name="uplo"> 'U' if upper triangle is stored in A, 'L' if lower triangle is stored </param>
        /// <returns></returns>
        public double[] EigenVectorsSymetric(double[] a, int n, int k, char uplo)
        {
            char jobz = 'V'; // store eigenvectors as well
            char range = 'I'; // to give a range of eigenvalues to return
            var w = new double[n]; // will contain eigenvalues of the matrix
            var z = new double[n * n]; // will contains eigenvectors of M
            var isuppz = new int[2 * k];
            var m = k;
            dsyevr(jobz, range, uplo, n, a, n, 0, 0, n - k - 1, n, 0, ref m, w, z, n, isuppz);
            return z;
        }

        /// <summary>
        /// Compute and return the kth first eigenvectors of a single precision symetric matrix.
        /// Matrix A should be stored in a row major fashion.
        /// </summary>
        /// <param name="a"> symetric matrix </param>
        /// <param name="n"> leading dimension of A </param>
        /// <param name="k"> number of eigen vectors to compute (between 1 and n) </param>
        /// <param name="uplo"> 'U' if upper triangle is stored in A, 'L' if lower triangle is stored </param>
        /// <returns></returns>
        public float[] EigenVectorsSymetric(float[] a, int n, int k, char uplo)
        {
            char jobz = 'V'; // store eigenvectors as well
            char range = 'I'; // to give a range of eigenvalues to return
            var w = new float[n]; // will contain eigenvalues of the matrix
            var z = new float[n * n]; // will contains eigenvectors of M
            var isuppz = new int[2 * k];
            var m = k;
            ssyevr(jobz, range, uplo, n, a, n, 0, 0, n - k - 1, n, 0, ref m, w, z, n, isuppz);
            return z;
        }

        /// <summary>
        /// Performs Singular Value Decomposition 
        /// on the given double precision general matrix
        /// Fill u and vt during the process
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="s"> </param>
        /// <param name="u"> left matrix filled during process </param>
        /// <param name="vt"> right matrix filled during process </param>
        public void SVD(double[] a, int m, int n, double[] s, double[] u, double[] vt)
        {
            double[] superb = new double[m - 1];
            dgesvd('A', 'A', m, n, a, n, s, u, m, vt, n, superb);
        }

        /// <summary>
        /// Performs Singular Value Decomposition 
        /// on the given single precision general matrix
        /// Fill u and vt during the process
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="s"> </param>
        /// <param name="u"> left matrix filled during process </param>
        /// <param name="vt"> right matrix filled during process </param>
        public void SVD(float[] a, int m, int n, float[] s, float[] u, float[] vt)
        {
            float[] superb = new float[m - 1];
            sgesvd('A', 'A', m, n, a, n, s, u, m, vt, n, superb);
        }
    }
}
