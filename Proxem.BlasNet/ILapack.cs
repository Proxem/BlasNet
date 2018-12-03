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
using System.Runtime.CompilerServices;
using System.Text;

namespace Proxem.BlasNet
{
    /// <summary>
    /// Lapack stands for Linear Algebra PACK.
    /// It contains high level linear algebra functions on matrices such as SVD decomposition,
    /// eigendecomposition,...
    /// This interface allows to wrap different Lapack implementations (principally for use in Windows and Linux).
    /// </summary>
    public interface ILapack
    {
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
        void sspsv(char uplo, int n, int nrhs, float[] a, int[] ipiv, float[] b, int ldb);

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
        void dspsv(char uplo, int n, int nrhs, double[] a, int[] ipiv, double[] b, int ldb);
        
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
        void dtrtrs(char uplo, char trans, char diag, int n, int nrhs, double[] a, int lda, double[] b, int ldb);

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
        void strtrs(char uplo, char trans, char diag, int n, int nrhs, float[] a, int lda, float[] b, int ldb);

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
        void dgesv(int n, int nrhs, double[] a, int lda, int[] ipiv, double[] b, int ldb);

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
        void sgesv(int n, int nrhs, float[] a, int lda, int[] ipiv, float[] b, int ldb);

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
        void dgelss(int m, int n, int nrhs, double[] a, int lda, double[] b, int ldb, double[] s, float rcond, ref int rank);

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
        void sgelss(int m, int n, int nrhs, float[] a, int lda, float[] b, int ldb, float[] s, float rcond, ref int rank);

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
        void dgels(char trans, int m, int n, int nrhs, double[] a, int lda, double[] b, int ldb);

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
        void sgels(char trans, int m, int n, int nrhs, float[] a, int lda, float[] b, int ldb);

        /// <summary>
        /// computes the singular value decomposition (SVD) 
        /// of a general rectangular single precision matrix A,
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
        /// 'A' (all m col of V^T are returned in vt), 
        /// 'S' (min(m,n) col of V^T are returned in vt), 
        /// 'O' (min(m,n) col of V^T overwritten in a),
        /// or 'N' (no col of V^T are computed)
        /// </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="a"> matrix a</param>
        /// <param name="lda"> leading dim of a </param>
        /// <param name="s"></param>
        /// <param name="u"> left matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U</param>
        /// <param name="vt"> right matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V </param>
        /// <param name="superb"> temporary storage </param>
        void sgesvd(char jobu, char jobvt, int m, int n, float[] a, int lda, float[] s, float[] u, int ldu, float[] vt, int ldvt, float[] superb);

        /// <summary>
        /// computes the singular value decomposition (SVD) 
        /// of a general rectangular double precision matrix A,
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
        /// 'A' (all m col of V^T are returned in vt), 
        /// 'S' (min(m,n) col of V^T are returned in vt), 
        /// 'O' (min(m,n) col of V^T overwritten in a),
        /// or 'N' (no col of V^T are computed)
        /// </param>
        /// <param name="m"> number of rows of a </param>
        /// <param name="n"> number of columns of a </param>
        /// <param name="a"> matrix a</param>
        /// <param name="lda"> leading dim of a </param>
        /// <param name="s"></param>
        /// <param name="u"> left matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U</param>
        /// <param name="vt"> right matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V </param>
        /// <param name="superb"> temporary storage </param>
        void dgesvd(char jobu, char jobvt, int m, int n, double[] a, int lda, double[] s, double[] u, int ldu, double[] vt, int ldvt, double[] superb);

        /// <summary>
        /// Singular Value Decomposition for general double precision rectangular matrix
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
        /// <param name="s"> temporary storage </param>
        /// <param name="u"> U matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U </param>
        /// <param name="vt"> V^T matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V^T </param>
        void dgesdd(char jobz, int m, int n, double[] a, int lda, double[] s, double[] u, int ldu, double[] vt, int ldvt);

        /// <summary>
        /// Singular Value Decomposition for general single precision rectangular matrix
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
        /// <param name="s"> temporary storage </param>
        /// <param name="u"> U matrix of SVD </param>
        /// <param name="ldu"> leading dimension of U </param>
        /// <param name="vt"> V^T matrix of SVD </param>
        /// <param name="ldvt"> leading dimension of V^T </param>
        void sgesdd(char jobz, int m, int n, float[] a, int lda, float[] s, float[] u, int ldu, float[] vt, int ldvt);

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
        void sgetrf(int m, int n, float[] a, int lda, int[] ipiv);

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
        void dgetrf(int m, int n, double[] a, int lda, int[] ipiv);

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
        void sgetri(int n, float[] a, int lda, int[] ipiv);

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
        void dgetri(int n, double[] a, int lda, int[] ipiv);

        /// <summary>
        /// Cholesky decomposition (double precision array)
        /// </summary>
        /// <param name="uplo"> 'U' if upper triangle is stored in a, else 'L' </param>
        /// <param name="n"> second dimension of a </param>
        /// <param name="a"> symetric matrix </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="offset"> starting offset in a </param>
        void dpotrf(char uplo, int n, double[] a, int lda, int offset = 0);

        /// <summary>
        /// Cholesky decomposition (single precision array)
        /// </summary>
        /// <param name="uplo"> 'U' if upper triangle is stored in a, else 'L' </param>
        /// <param name="n"> second dimension of a </param>
        /// <param name="a"> symetric matrix </param>
        /// <param name="lda"> leading dimension of a </param>
        /// <param name="offset"> starting offset in a </param>
        void spotrf(char uplo, int n, float[] a, int lda, int offset = 0);

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
        void dsyev(char jobz, char uplo, int n, double[] a, int lda, double[] w);

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
        void ssyev(char jobz, char uplo, int n, float[] a, int lda, float[] w);

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
        void dsyevr(int jobz, char range, char uplo, int n, double[] a, int lda, double vl, double vu, int il, int iu, float abstol, ref int m, double[] w, double[] z, int ldz, int[] isuppz);

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
        void ssyevr(int jobz, char range, char uplo, int n, float[] a, int lda, float vl, float vu, int il, int iu, float abstol, ref int m, float[] w, float[] z, int ldz, int[] isuppz);

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
        void dgeev(char jobvl, char jobvr, int n, double[] a, int lda, double[] wr, double[] wi, double[] vl, int ldvl, double[] vr, int ldvr);

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
        void sgeev(char jobvl, char jobvr, int n, float[] a, int lda, float[] wr, float[] wi, float[] vl, int ldvl, float[] vr, int ldvr);

        // High level functions 

        /// <summary>
        /// Returns the determinant of a given general single precision square matrix of order n
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        float Determinant(float[] a, int n);

        /// <summary>
        /// Returns the determinant of a given general double precision square matrix of order n
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        double Determinant(double[] a, int n);

        /// <summary>
        /// return the determinant of a single precision square matrix
        /// represented as a 2D C# array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> the ordrer of a </param>
        /// <returns> determinant of a </returns>
        float Determinant(float[,] a, int n);

        /// <summary>
        /// return the determinant of a double precision matrix
        /// represented as a 2D C# array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> the ordrer of a </param>
        /// <returns> determinant of a </returns>
        double Determinant(double[,] a, int n);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="n"></param>
        /// <param name="offset"></param>
        /// <param name="lda"></param>
        /// <returns></returns>
        Tuple<float, float> SLogDet(float[] a, int n, int offset = 0, int lda = 0);

        /// <summary>
        /// Inverse inplace a given single precision matrix represented as a 2D array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        void Inverse(float[,] a, int n);

        /// <summary>
        /// Inverse inplace a given double precision matrix represented as a 2D array
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        void Inverse(double[,] a, int n);

        /// <summary>
        /// Inverse inplace a single precision matrix 
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <param name="offset"> starting offset of a </param>
        /// <param name="lda"> leading dimension of a</param>
        void Inverse(float[] a, int n, int offset = 0, int lda = 0);

        /// <summary>
        /// Inverse inplace a double precision matrix 
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <param name="offset"> starting offset of a </param>
        /// <param name="lda"> leading dimension of a</param>
        void Inverse(double[] a, int n, int offset = 0, int lda = 0);

        /// <summary>
        /// Inverse single precision matrix inplace
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        void Inverse(double[] a, int n);

        /// <summary>
        /// Inverse double precision matrix inplace
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        void Inverse(float[] a, int n);

        /// <summary>
        /// Inverse double matrix precision inplace
        /// </summary>
        /// <param name="a"></param>
        /// <param name="n"></param>
        /// <param name="m"></param>
        void Inverse(double[,] a, int n, int m);

        /// <summary>
        /// Return the list of eigenvalues of a double precision symetric matrix.
        /// Matrix should be store in row major fashion and only upper triangle need be stored
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        double[] EigenValuesSymetric(double[] a, int n);

        /// <summary>
        /// Return the list of eigenvalues of a single precision symetric matrix.
        /// Matrix should be store in row major fashion and only upper triangle need be stored
        /// </summary>
        /// <param name="a"> matrix a </param>
        /// <param name="n"> order of a </param>
        /// <returns>list of eigenvalues</returns>
        float[] EigenValuesSymetric(float[] a, int n);

        /// <summary>
        /// Compute and return the kth first eigenvectors of a double precision symetric matrix.
        /// Matrix A should be stored in a row major fashion.
        /// </summary>
        /// <param name="a"> symetric matrix </param>
        /// <param name="n"> leading dimension of A </param>
        /// <param name="k"> number of eigen vectors to compute (between 1 and n) </param>
        /// <param name="uplo"> 'U' if upper triangle is stored in A, 'L' if lower triangle is stored </param>
        /// <returns>list of eigenvalues</returns>
        double[] EigenVectorsSymetric(double[] a, int n, int k, char uplo);

        /// <summary>
        /// Compute and return the kth first eigenvectors of a single precision symetric matrix.
        /// Matrix A should be stored in a row major fashion.
        /// </summary>
        /// <param name="a"> symetric matrix </param>
        /// <param name="n"> leading dimension of A </param>
        /// <param name="k"> number of eigen vectors to compute (between 1 and n) </param>
        /// <param name="uplo"> 'U' if upper triangle is stored in A, 'L' if lower triangle is stored </param>
        /// <returns></returns>
        float[] EigenVectorsSymetric(float[] a, int n, int k, char uplo);

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
        void SVD(double[] a, int m, int n, double[] s, double[] u, double[] vt);

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
        void SVD(float[] a, int m, int n, float[] s, float[] u, float[] vt);
    }

    /// <summary>
    /// Static wrapper around an <see cref="ILapack"> implementation </see>.
    /// All methods of NumNet calling Lapack will use this facade.
    /// </summary>
    public static class Lapack
    {
        /// <summary> The actual provider. </summary>
        public static ILapack Provider;

        /// <summary>
        /// Computes the solution to the system of linear equations 
        /// with a real or complex symmetric single precision matrix A stored in packed format,
        /// and multiple right-hand sides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spsv(char uplo, int n, int nrhs, float[] a, int[] ipiv, float[] b, int ldb)
        {
            Provider.sspsv(uplo, n, nrhs, a, ipiv, b, ldb);
        }

        /// <summary>
        /// Computes the solution to the system of linear equations 
        /// with a real or complex symmetric double precision matrix A stored in packed format,
        /// and multiple right-hand sides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spsv(char uplo, int n, int nrhs, double[] a, int[] ipiv, double[] b, int ldb)
        {
            Provider.dspsv(uplo, n, nrhs, a, ipiv, b, ldb);
        }

        /// <summary>
        /// solves a system of linear equation with triangular single precision matrix
        /// find X s.t. A.X = B
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void trtrs(char uplo, char trans, char diag, int n, int nrhs, float[] a, int lda, float[] b, int ldb)
        {
            Provider.strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
        }

        /// <summary>
        /// solves a system of linear equation with triangular double precision matrix
        /// find X s.t. A.X = B
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void trtrs(char uplo, char trans, char diag, int n, int nrhs, double[] a, int lda, double[] b, int ldb)
        {
            Provider.dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
        }

        /// <summary>
        /// solve a system of linear equation with general double precision square matrix
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gesv(int n, int nrhs, double[] a, int lda, int[] ipiv, double[] b, int ldb)
        {
            Provider.dgesv(n, nrhs, a, lda, ipiv, b, ldb);
        }

        /// <summary>
        /// solve a system of linear equation with general single precision square matrix
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gesv(int n, int nrhs, float[] a, int lda, int[] ipiv, float[] b, int ldb)
        {
            Provider.sgesv(n, nrhs, a, lda, ipiv, b, ldb);
        }

        /// <summary>
        /// Compute the minimum norm solution to a double precision linear least squares problem
        /// Uses SVD decomposition
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gelss(int m, int n, int nrhs, double[] a, int lda, double[] b, int ldb, double[] s, float rcond, ref int rank)
        {
            Provider.dgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, ref rank);
        }

        /// <summary>
        /// Compute the minimum norm solution to a single precision linear least squares problem
        /// Uses SVD decomposition
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gelss(int m, int n, int nrhs, float[] a, int lda, float[] b, int ldb, float[] s, float rcond, ref int rank)
        {
            Provider.sgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, ref rank);
        }

        /// <summary>
        ///  Uses QR or LQ factorization to solve an overdetermined or underdetermined linear
        ///  system with full rank double precision matrix a.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gels(char trans, int m, int n, int nrhs, double[] a, int lda, double[] b, int ldb)
        {
            Provider.dgels(trans, m, n, nrhs, a, lda, b, ldb);
        }

        /// <summary>
        ///  Uses QR or LQ factorization to solve an overdetermined or underdetermined linear
        ///  system with full rank single precision matrix a.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gels(char trans, int m, int n, int nrhs, float[] a, int lda, float[] b, int ldb)
        {
            Provider.sgels(trans, m, n, nrhs, a, lda, b, ldb);
        }

        /// <summary>
        /// computes the singular value decomposition (SVD) 
        /// of a general rectangular single precision matrix A,
        /// optionally the left and/or right singular vectors.
        /// A = U * Sigma * V^T 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gesvd(char jobu, char jobvt, int m, int n, float[] a, int lda, float[] s, float[] u, int ldu, float[] vt, int ldvt, float[] superb)
        {
            Provider.sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
        }

        /// <summary>
        /// computes the singular value decomposition (SVD) 
        /// of a general rectangular double precision matrix A,
        /// optionally the left and/or right singular vectors.
        /// A = U * Sigma * V^T 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gesvd(char jobu, char jobvt, int m, int n, double[] a, int lda, double[] s, double[] u, int ldu, double[] vt, int ldvt, double[] superb)
        {
            Provider.dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
        }

        /// <summary>
        /// Singular Value Decomposition for general double precision rectangular matrix
        /// A = U*D*V^T
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gesdd(char jobz, int m, int n, double[] a, int lda, double[] s, double[] u, int ldu, double[] vt, int ldvt)
        {
            Provider.dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
        }

        /// <summary>
        /// Singular Value Decomposition for general single precision rectangular matrix
        /// A = U*D*V^T
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void gesdd(char jobz, int m, int n, float[] a, int lda, float[] s, float[] u, int ldu, float[] vt, int ldvt)
        {
            Provider.sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
        }

        /// <summary>
        /// The routine computes the LU factorization 
        /// of a general m-by-n single precision matrix A as
        /// A = P*L*U,
        /// where P is a permutation matrix, L is lower triangular with unit diagonal elements(lower trapezoidal if m > n)
        /// and U is upper triangular(upper trapezoidal if m &lt; n). The routine uses partial pivoting, with row interchanges.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void getrf(int m, int n, float[] a, int lda, int[] ipiv)
        {
            Provider.sgetrf(m, n, a, lda, ipiv);
        }

        /// <summary>
        /// The routine computes the LU factorization 
        /// of a general m-by-n double precision matrix A as
        /// A = P*L*U,
        /// where P is a permutation matrix, L is lower triangular with unit diagonal elements(lower trapezoidal if m > n)
        /// and U is upper triangular(upper trapezoidal if m &lt; n). The routine uses partial pivoting, with row interchanges.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void getrf(int m, int n, double[] a, int lda, int[] ipiv)
        {
            Provider.dgetrf(m, n, a, lda, ipiv);
        }

        /// <summary>
        /// Computes the inverse of an LU-factored general single precision matrix.
        /// The routine computes the inverse inv(A) of a general matrix A. Before calling this routine, call sgetrf to factorize A.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void getri(int n, float[] a, int lda, int[] ipiv)
        {
            Provider.sgetri(n, a, lda, ipiv);
        }

        /// <summary>
        /// Computes the inverse of an LU-factored general double precision matrix.
        /// The routine computes the inverse inv(A) of a general matrix A. Before calling this routine, call sgetrf to factorize A.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void getri(int n, double[] a, int lda, int[] ipiv)
        {
            Provider.dgetri(n, a, lda, ipiv);
        }

        /// <summary>
        /// Cholesky decomposition (double precision array)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void potrf(char uplo, int n, double[] a, int lda, int offset = 0)
        {
            Provider.dpotrf(uplo, n, a, lda, offset);
        }

        /// <summary>
        /// Cholesky decomposition (single precision array)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void potrf(char uplo, int n, float[] a, int lda, int offset = 0)
        {
            Provider.spotrf(uplo, n, a, lda, offset);
        }

        /// <summary>
        /// Compute all eigenvalues and optionnaly eigenvectors
        /// of a double precision matrix a
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void syev(char jobz, char uplo, int n, double[] a, int lda, double[] w)
        {
            Provider.dsyev(jobz, uplo, n, a, lda, w);
        }

        /// <summary>
        /// Compute all eigenvalues and optionnaly eigenvectors
        /// of a single precision matrix a
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void syev(char jobz, char uplo, int n, float[] a, int lda, float[] w)
        {
            Provider.ssyev(jobz, uplo, n, a, lda, w);
        }

        /// <summary>
        /// Compute selected eigenvalues and optionnally eigenvectors 
        /// of a double precision symetric matrix a
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void syevr(int jobz, char range, char uplo, int n, double[] a, int lda, double vl, double vu, int il, int iu, float abstol, ref int m, double[] w, double[] z, int ldz, int[] isuppz)
        {
            Provider.dsyevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, ref m, w, z, ldz, isuppz);
        }

        /// <summary>
        /// Compute selected eigenvalues and optionnally eigenvectors 
        /// of a single precision symetric matrix a
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void syevr(int jobz, char range, char uplo, int n, float[] a, int lda, float vl, float vu, int il, int iu, float abstol, ref int m, float[] w, float[] z, int ldz, int[] isuppz)
        {
            Provider.ssyevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, ref m, w, z, ldz, isuppz);
        }

        /// <summary>
        /// Compute eigenvalues and optionnally eigenvectors
        /// of a general double precision square matrix
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void geev(char jobvl, char jobvr, int n, double[] a, int lda, double[] wr, double[] wi, double[] vl, int ldvl, double[] vr, int ldvr)
        {
            Provider.dgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
        }

        /// <summary>
        /// Compute eigenvalues and optionnally eigenvectors
        /// of a general single precision square matrix
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void geev(char jobvl, char jobvr, int n, float[] a, int lda, float[] wr, float[] wi, float[] vl, int ldvl, float[] vr, int ldvr)
        {
            Provider.sgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
        }


        // High level functions 

        /// <summary>
        /// Returns the determinant of a given general single precision square matrix of order n
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Determinant(float[] a, int n)
        {
            return Provider.Determinant(a, n);
        }

        /// <summary>
        /// Returns the determinant of a given general double precision square matrix of order n
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Determinant(double[] a, int n)
        {
            return Provider.Determinant(a, n);
        }

        /// <summary>
        /// return the determinant of a single precision square matrix
        /// represented as a 2D C# array
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Determinant(float[,] a, int n)
        {
            return Provider.Determinant(a, n);
        }

        /// <summary>
        /// return the determinant of a double precision square matrix
        /// represented as a 2D C# array
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Determinant(double[,] a, int n)
        {
            return Provider.Determinant(a, n);
        }

        /// <summary>
        /// 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Tuple<float, float> SLogDet(float[] a, int n, int offset = 0, int lda = 0)
        {
            return Provider.SLogDet(a, n, offset, lda);
        }

        /// <summary>
        /// Inverse inplace a given single precision matrix represented as a 2D array
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Inverse(float[,] a, int n)
        {
            Provider.Inverse(a, n);
        }

        /// <summary>
        /// Inverse inplace a given double precision matrix represented as a 2D array
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Inverse(double[,] a, int n)
        {
            Provider.Inverse(a, n);
        }

        /// <summary>
        /// Inverse inplace a single precision matrix 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Inverse(float[] a, int n, int offset = 0, int lda = 0)
        {
            Provider.Inverse(a, n, offset, lda);
        }

        /// <summary>
        /// Inverse inplace a single precision matrix 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Inverse(double[] a, int n, int offset = 0, int lda = 0)
        {
            Provider.Inverse(a, n, offset, lda);
        }

        /// <summary>
        /// Inverse inplace a single precision matrix 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Inverse(double[] a, int n)
        {
            Provider.Inverse(a, n);
        }

        /// <summary>
        /// Inverse inplace a single precision matrix 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Inverse(float[] a, int n)
        {
            Provider.Inverse(a, n);
        }

        /// <summary>
        /// Inverse inplace a single precision matrix 
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Inverse(double[,] a, int n, int m)
        {
            Provider.Inverse(a, n, m);
        }

        /// <summary>
        /// Return the list of eigenvalues of a double precision symetric matrix.
        /// Matrix should be store in row major fashion and only upper triangle need be stored
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double[] EigenValuesSymetric(double[] a, int n)
        {
            return Provider.EigenValuesSymetric(a, n);
        }

        /// <summary>
        /// Return the list of eigenvalues of a single precision symetric matrix.
        /// Matrix should be store in row major fashion and only upper triangle need be stored
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] EigenValuesSymetric(float[] a, int n)
        {
            return Provider.EigenValuesSymetric(a, n);
        }

        /// <summary>
        /// Compute and return the kth first eigenvectors of a double precision symetric matrix.
        /// Matrix A should be stored in a row major fashion.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double[] EigenVectorsSymetric(double[] a, int n, int k, char uplo)
        {
            return EigenVectorsSymetric(a, n, k, uplo);
        }

        /// <summary>
        /// Compute and return the kth first eigenvectors of a single precision symetric matrix.
        /// Matrix A should be stored in a row major fashion.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float[] EigenVectorsSymetric(float[] a, int n, int k, char uplo)
        {
            return EigenVectorsSymetric(a, n, k, uplo);
        }

        /// <summary>
        /// Performs Singular Value Decomposition 
        /// on the given double precision general matrix
        /// Fill u and vt during the process
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SVD(double[] a, int m, int n, double[] s, double[] u, double[] vt)
        {
            Provider.SVD(a, m, n, s, u, vt);
        }

        /// <summary>
        /// Performs Singular Value Decomposition 
        /// on the given single precision general matrix
        /// Fill u and vt during the process
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SVD(float[] a, int m, int n, float[] s, float[] u, float[] vt)
        {
            Provider.SVD(a, m, n, s, u, vt);
        }
    }
}
