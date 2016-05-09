/**
 * \file fl/detail/lsq.h
 *
 * \brief Least-squares problem solvers.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2014 Marco Guazzone (marco.guazzone@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FL_DETAIL_LSQ_H
#define FL_DETAIL_LSQ_H


#include <cmath>
#include <cstddef>
#include <fl/macro.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef FLX_CONFIG_HAVE_LAPACK
# include <algorithm>
# ifdef FLX_CONFIG_HAVE_LAPACKE
#  include <lapacke.h>
# endif // FLX_CONFIG_HAVE_LAPACKE
# include <sstream>
#endif // FLX_CONFIG_HAVE_LAPACK


namespace fl { namespace detail {

/////////////////
// Declarations
/////////////////

template <typename ValueT, typename MatrixT, typename VectorT>
std::vector<ValueT> LsqSolve(const MatrixT& A, const VectorT& b);

template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMulti(const AMatrixT& A, const BMatrixT& B);


/////////////////
// Definitions
/////////////////

#ifdef FLX_CONFIG_HAVE_LAPACK

#ifndef FLX_CONFIG_HAVE_LAPACKE

typedef int lapack_int;
typedef int lapack_logical;

extern "C"
void dgecon_(const char* norm, const lapack_int* n, const double* a,
             const lapack_int* lda, const double* anorm, double* rcond,
             double* work, lapack_int* iwork, lapack_int* info);

extern "C"
void dgels_(const char* trans, const lapack_int* m,
            const lapack_int* n, const lapack_int* nrhs, double* a,
            const lapack_int* lda, double* b, const lapack_int* ldb,
            double* work, const lapack_int* lwork, lapack_int* info);

extern "C"
void dgelsd_(const lapack_int* m, const lapack_int* n,
             const lapack_int* nrhs, double* a, const lapack_int* lda,
             double* b, const lapack_int* ldb, double* s, const double* rcond,
             lapack_int* rank, double* work, const lapack_int* lwork,
             lapack_int* iwork, lapack_int* info);

extern "C"
void dgelss_(const lapack_int* m, const lapack_int* n,
             const lapack_int* nrhs, double* a, const lapack_int* lda,
             double* b, const lapack_int* ldb, double* s, const double* rcond,
             lapack_int* rank, double* work, const lapack_int* lwork,
             lapack_int* info);

extern "C"
void dgelsy_(const lapack_int* m, const lapack_int* n,
             const lapack_int* nrhs, double* a, const lapack_int* lda,
             double* b, const lapack_int* ldb, lapack_int* jpvt,
             const double* rcond, lapack_int* rank, double* work,
             const lapack_int* lwork, lapack_int* info);

extern "C"
void dgeqrf_(const lapack_int* m, const lapack_int* n, double* a,
             const lapack_int* lda, double* tau, double* work,
             const lapack_int* lwork, lapack_int* info);

extern "C"
void dgesvd_(const char* jobu, const char* jobvt,
             const lapack_int* m, const lapack_int* n, double* a,
             const lapack_int* lda, double* s, double* u,
             const lapack_int* ldu, double* vt, const lapack_int* ldvt,
             double* work, const lapack_int* lwork, lapack_int* info );

extern "C"
void dgetrf_(const lapack_int* m, const lapack_int* n, double* a,
        const lapack_int* lda, lapack_int* ipiv, lapack_int* info);

extern "C"
double dlange_(const char* norm, const lapack_int* m,
               const lapack_int* n, const double* a, const lapack_int* lda,
               double* work );

extern "C"
lapack_int ilaenv_(const lapack_int* ispec, const char* name,
                   const char* opt, const lapack_int* n1, const lapack_int* n2,
                   const lapack_int* n3, const lapack_int* n4);

#endif // FLX_CONFIG_HAVE_LAPACKE

/// Fortran BLAS DGEMM subroutine
extern "C"
void dgemm_(const char* transa, const char* transb,
            const lapack_int* m, const lapack_int* n, const lapack_int* k,
            const double* alpha, double* A, const lapack_int* lda,
            double* B, const lapack_int* ldb, const double* beta,
            double* C, const lapack_int* ldc);

///// Fortran BLAS DNRM2 subrouting
//extern "C"
//void dnrm2_(const lapack_int* n, double* x, const lapack_int* incx, double* norm);


static double* LapackLsqSolveGELS(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs);
static double* LapackLsqSolveGELSD(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs);
static double* LapackLsqSolveGELSS(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs);
static double* LapackLsqSolveGELSY(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs);
static double* LapackLsqResidualsNorm(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs, lapack_int ldB, double* X, lapack_int ldX);
static double LapackMatrixRCond(double* A, lapack_int m, lapack_int n, lapack_int ldA);
template <typename ValueT>
lapack_logical LapackMatrixIsSingular(const ValueT* flat_A, lapack_int A_m, lapack_int A_n, lapack_int ldA);
template <typename ValueT>
lapack_int LapackMatrixRank(const ValueT* flat_A, lapack_int A_m, lapack_int A_n, lapack_int ldA);
template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMultiQR(const AMatrixT& A, const BMatrixT& B);
template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMultiSVD(const AMatrixT& A, const BMatrixT& B);
template <typename ValueT, typename MatrixT, typename VectorT>
std::vector<ValueT> LsqSolve(const MatrixT& A, const VectorT& b);
template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMulti(const AMatrixT& A, const BMatrixT& B);


template <typename ValueT>
lapack_int LapackMatrixRank(const ValueT* flat_A, lapack_int A_m, lapack_int A_n, lapack_int ldA)
{
    const ValueT eps = std::numeric_limits<ValueT>::epsilon();
    const lapack_int k = std::min(A_m, A_n);

    lapack_int info = 0;

    ValueT* s = 0;
    ValueT* superb = 0;

    ValueT U = 0;
    ValueT V = 0;

    lapack_int rank = 0;

    try
    {
        s = new ValueT[k];
        std::fill(s, s+k, 0);

        superb = new ValueT[k-1];
        std::fill(s, s+k, 0);

#ifdef FLX_CONFIG_HAVE_LAPACKE
        info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', A_m, A_n, flat_A, ldA, s, &U, 1, &V, 1, superb);
#else
        {
            const char jobu = 'N';
            const char jobvt = 'N';
            const lapack_int ldU = 1;
            const lapack_int ldV = 1;
            const lapack_int ldVT = 1;
            const lapack_int lwork = std::max(1,5*std::min(A_m,A_n));
            double work[lwork];
            dgesvd_(&jobu, &jobvt, &A_m, &A_n, flat_A, &ldA, &s, &U, &ldU, &V, &ldV, superb, &ldVT, &work, &lwork, &info);
        }
#endif // FLX_CONFIG_HAVE_LAPACKE
        if (info)
        {
            std::ostringstream oss;
            oss << "Error during rank computation (GESVD error code: " << info << ")";
            FL_THROW2(std::runtime_error, oss.str());
        }

        ValueT norm2 = VectorMax(s, A_n);

        ValueT* tol = std::max(A_m, A_n)*eps*norm2;
        lapack_int rank = 0;
        for (lapack_int i = 0; i < A_n; ++i)
        {
            if (s[i] > tol)
            {
                ++rank;
            }
        }
    }
    catch(...)
    {
        if (s)
        {
            delete[] s;
        }
        if (superb)
        {
            delete[] superb;
        }

        throw;
    }

    if (s)
    {
        delete[] s;
    }
    if (superb)
    {
        delete[] superb;
    }

    return rank;
}

template <typename ValueT>
lapack_logical LapackMatrixIsSingular(const ValueT* flat_A, lapack_int A_m, lapack_int A_n, lapack_int ldA)
{
    lapack_int rank = LapackMatrixRank(flat_A, A_m, A_n, ldA);
    if (rank < std::min(A_m, A_n))
    {
        return 1;
    }
    return 0;
}

double LapackMatrixRCond(double* A, lapack_int m, lapack_int n, lapack_int ldA)
{
	assert( A );
	assert( m > 0 );
	assert( n > 0 );
	assert( ldA > 0 );

	double rc = 0; // The estimation of reciprocal condition number of A

	lapack_int k = std::min(m, n);
	lapack_int info = 0;

	if (m == n)
	{
		// Square matrix -> use A directly

		// Compute the norm-1 of A
		double norm = 0;
#ifdef FLX_CONFIG_HAVE_LAPACKE
		norm = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', m, n, A, ldA);
#else
        {
            const char norm_type = '1';
            double work = 0;
            norm = dlange_(&norm_type, &m, &n, A, &ldA, &work);
        }
#endif // FLX_CONFIG_HAVE_LAPACKE

		lapack_int* ipiv = new lapack_int[k];
		std::fill(ipiv, ipiv+k, 0);

		// Perform LU factorization

		// Make a copy of A to avoid to change its content
		lapack_int A_sz = m*n;
		double* AA = new double[m*n];
		std::copy(A, A+A_sz, AA);

#ifdef FLX_CONFIG_HAVE_LAPACKE
		info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, AA, ldA, ipiv);
#else
        dgetrf_(&m, &n, A, &ldA, ipiv, &info);
#endif // FLX_CONFIG_HAVE_LAPACKE
		if (info)
		{
			if (!AA)
			{
				delete[] AA;
			}
			if (!ipiv)
			{
				delete[] ipiv;
			}
			std::ostringstream oss;
			oss << "Error during LU decomposition. LAPACK xGETRF returned: " << info << ".";
			throw std::runtime_error(oss.str());
		}
		if (!AA)
		{
			AA = 0;
			delete[] AA;
		}
		if (!ipiv)
		{
			ipiv = 0;
			delete[] ipiv;
		}

		// Estimate the reciprocal condition number
#ifdef FLX_CONFIG_HAVE_LAPACKE
		info = LAPACKE_dgecon(LAPACK_COL_MAJOR, '1', n, A, ldA, norm, &rc);
#else
        {
            const char norm_type = '1';
            const lapack_int lwork = std::max(1,4*n);
            double work[lwork];
            const lapack_int liwork = std::max(1,n);
            lapack_int iwork[liwork];
            dgecon_(&norm_type, &n, A, &ldA, &norm, &rc, work, iwork, &info);
        }
#endif // FLX_CONFIG_HAVE_LAPACKE
		if (info)
		{
			std::ostringstream oss;
			oss << "Error during the estimation of reciprocal condition number. LAPACK xGECON returned: " << info << ".";
			throw std::runtime_error(oss.str());
		}
	}
	else
	{
		// Rectangular matrix -> use the QR factorization of A

		lapack_int A_sz = m*n;

    	double* QR = new double[A_sz];
		lapack_int QR_m = 0;
		lapack_int QR_n = 0;

		if (m > n)
		{
			// # rows > # cols -> QR-factorize A

			std::copy(A, A+A_sz, QR);

			QR_m = m;
			QR_n = n;
		}
		else
		{
			// # rows < # cols -> QR-factorize the transpose of A

			// Copy the transpose of A in QR
			for (lapack_int j = 0; j < m; ++j)
			{
				const lapack_int offs = j*n;
				for (lapack_int i = 0; i < n; ++i)
				{
					QR[offs+i] = A[j+i*m];
				}
			}

			QR_m = n;
			QR_n = m;
		}

		// Perform the QR factorization

		lapack_int ldQR = QR_m;

		double* tau = new double[k];
		std::fill(tau, tau+k, 0);

#ifdef FLX_CONFIG_HAVE_LAPACKE
		info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, QR_m, QR_n, QR, ldQR, tau);
#else
        {
            const lapack_int lwork = std::max(1,n);
            double work[lwork];
            dgeqrf_(&m, &n, A, &ldA, tau, work, &lwork, &info);
        }
#endif // FLX_CONFIG_HAVE_LAPACKE
		if (info)
		{
			if (QR)
			{
				delete[] QR;
			}
			if (tau)
			{
				delete[] tau;
			}

			std::ostringstream oss;
			oss << "Error during QR factorization. LAPACK xGEQRF returned: " << info << ".";
			throw std::runtime_error(oss.str());
		}
		if (tau)
		{
			tau = 0;
			delete[] tau;
		}

		// Extract the R matrix

		double* R = new double[k*QR_n];
		lapack_int R_m = k;
		lapack_int R_n = QR_n;
		lapack_int ldR = k;

		for (lapack_int i = 0; i < R_m; ++i)
		{
			for (lapack_int j = 0; j < R_n; ++j)
			{
				if (j >= i)
				{
					R[i+ldR*j] = QR[i+ldQR*j];
				}
				else
				{
					R[i+ldR*j] = 0;
				}
			}
		}

		if (QR)
		{
			QR = 0;
			delete[] QR;
		}

		// Estimate the reciprocal condition number of R
		try
		{
			rc = LapackMatrixRCond(R, R_m, R_n, ldR);
		}
		catch(...)
		{
			if (R)
			{
				delete[] R;
			}
			throw;
		}
		if (R)
		{
			R = 0;
			delete[] R;
		}
	}

	return rc;
}

double* LapackLsqResidualsNorm(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs, lapack_int ldB, double* X, lapack_int ldX)
{
	double* norm = 0;

	double dbl_one = 1;
	double dbl_minus_one = -1;
	//lapack_int int_one = 1;
	char notrans = 'N';

	// Make a copy of B to avoid to change it directly
	lapack_int B_sz = m*nrhs;
	double* R = new double[B_sz];
	std::copy(B, B+B_sz, R);
	lapack_int ldR = ldB;

	// Compute R=B-AX
	dgemm_(&notrans, &notrans, &m, &nrhs, &n, &dbl_minus_one, A, &ldA, X, &ldX, &dbl_one, R, &ldR);

	// Compute ||R||
	norm = new double[nrhs];
	std::fill(norm, norm+nrhs, 0);
	for (lapack_int j = 0; j < nrhs; ++j)
	{
/*XXX don't work, why?!?
		dnrm2_(&m, &R[j*ldR], &int_one, &norm[j]);
*/
		for (lapack_int i = 0; i < m; ++i)
		{
			norm[j] += R[i+j*ldR]*R[i+j*ldR];
		}
		norm[j] = std::sqrt(norm[j]);
	}
	if (R)
	{
		delete[] R;
	}

	return norm;
}

double* LapackLsqSolveGELS(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs)
{
	assert( A );
	assert( m > 0 );
	assert( n > 0 );
	assert( ldA > 0 );
	assert( B );
	assert( nrhs > 0 );

	// Make a copy of the input matrix A to avoid changing its content
	lapack_int A_sz = m*n;
	double* AA = new double[A_sz];
	std::copy(A, A+A_sz, AA);

	const lapack_int ldX = std::max(m,n);

	// Make a copy of the input matrix B to avoid changing its content
	double* X = new double[nrhs*ldX];
	std::copy(B, B+m*nrhs, X);
	std::fill(X+m*nrhs, X+ldX*nrhs, 0);

	lapack_int info = 0;
#ifdef FLX_CONFIG_HAVE_LAPACKE
	info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, AA, ldA, X, ldX );
#else
    {
        const char trans = 'N';
        const lapack_int lwork = std::max(1, std::min(m,n)+std::max(std::min(m,n), nrhs));
        double work[lwork];
        dgels_(&trans, &m, &n, &nrhs, AA, &ldA, X, &ldX, work, &lwork, &info);
    }
#endif // FLX_CONFIG_HAVE_LAPACKE
	if (info)
	{
		if (AA)
		{
			delete[] AA;
		}
		if (X)
		{
			delete[] X;
		}

		std::ostringstream oss;
		oss << "Unable to solve LSQ problem. LAPACK xGELS returned: " << info;
		throw std::runtime_error(oss.str());
	}
	if (AA)
	{
		AA = 0;
		delete[] AA;
	}

	return X;
}

double* LapackLsqSolveGELSD(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs)
{
	assert( A );
	assert( m > 0 );
	assert( n > 0 );
	assert( ldA > 0 );
	assert( B );
	assert( nrhs > 0 );

	const lapack_int ldX = std::max(m,n);

	double rc = LapackMatrixRCond(A, m, n, ldA);

	// Make a copy of the input matrix A to avoid changing its content
	lapack_int A_sz = m*n;
	double* AA = new double[A_sz];
	std::copy(A, A+A_sz, AA);

	// Make a copy of the input matrix B to avoid changing its content
	double* X = new double[n*ldX];
	std::copy(B, B+m*nrhs, X);
	std::fill(X+m*nrhs, X+n*ldX, 0);

	double* s = new double[m];

	lapack_int rank = 0;

	lapack_int info = 0;
#ifdef FLX_CONFIG_HAVE_LAPACKE
	info = LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, n, nrhs, AA, ldA, X, ldX, s, rc, &rank);
#else
    {
        const lapack_int ispec = 9;
        const char* name = "GELSD";
        const char* opts = "";
        const lapack_int n1 = 0;
        const lapack_int n2 = 0;
        const lapack_int n3 = 0;
        const lapack_int n4 = 0;
        const lapack_int smlsiz = ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4);
        const lapack_int minmn = std::min(m,n);
        const lapack_int nlvl = std::max(static_cast<lapack_int>(std::log(static_cast<double>(minmn)/static_cast<double>(smlsiz+1))/std::log(2.0)) + 1, 0);
        const lapack_int lwork = std::max(1, 12*minmn+2*minmn*smlsiz+8*minmn*nlvl+minmn*nrhs+(smlsiz+1)*(smlsiz+1));
        double work[lwork];
        const lapack_int liwork = std::max(1, 3*minmn*nlvl+11*minmn);
        lapack_int iwork[liwork];
        dgelsd_(&m, &n, &nrhs, AA, &ldA, X, &ldX, s, &rc, &rank, work, &lwork, iwork, &info);
    }
#endif // FLX_CONFIG_HAVE_LAPACKE
	if (info)
	{
		if (AA)
		{
			delete[] AA;
		}
		if (X)
		{
			delete[] X;
		}
		if (s)
		{
			delete[] s;
		}

		std::ostringstream oss;
		oss << "Unable to solve LSQ problem. LAPACK xGELSD returned: " << info;
		throw std::runtime_error(oss.str());
	}
	if (AA)
	{
		AA = 0;
		delete[] AA;
	}
	if (s)
	{
		s = 0;
		delete[] s;
	}

	return X;
}

double* LapackLsqSolveGELSS(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs)
{
	assert( A );
	assert( m > 0 );
	assert( n > 0 );
	assert( ldA > 0 );
	assert( B );
	assert( nrhs > 0 );

	const lapack_int ldX = std::max(m,n);

	double rc = LapackMatrixRCond(A, m, n, ldA);

	// Make a copy of the input matrix A to avoid changing its content
	lapack_int A_sz = m*n;
	double* AA = new double[A_sz];
	std::copy(A, A+A_sz, AA);

	// Make a copy of the input matrix B to avoid changing its content
	double* X = new double[n*ldX];
	std::copy(B, B+m*nrhs, X);
	std::fill(X+m*nrhs, X+n*ldX, 0);

	double* s = new double[m];

	lapack_int rank = 0;

	lapack_int info = 0;
#ifdef FLX_CONFIG_HAVE_LAPACKE
	info = LAPACKE_dgelss(LAPACK_COL_MAJOR, m, n, nrhs, AA, ldA, X, ldX, s, rc, &rank);
#else
    {
        const lapack_int minmn = std::min(m,n);
        const lapack_int lwork = std::max(1, 3*minmn+std::max(std::max(2*minmn, std::max(m,n)), nrhs));
        double work[lwork];
        dgelss_(&m, &n, &nrhs, AA, &ldA, X, &ldX, s, &rc, &rank, work, &lwork, &info);
    }
#endif // FLX_CONFIG_HAVE_LAPACKE
	if (info)
	{
		if (AA)
		{
			delete[] AA;
		}
		if (X)
		{
			delete[] X;
		}
		if (s)
		{
			delete[] s;
		}

		std::ostringstream oss;
		oss << "Unable to solve LSQ problem. LAPACK xGELSS returned: " << info;
		throw std::runtime_error(oss.str());
	}
	if (AA)
	{
		AA = 0;
		delete[] AA;
	}
	if (s)
	{
		s = 0;
		delete[] s;
	}

	return X;
}

double* LapackLsqSolveGELSY(double* A, lapack_int m, lapack_int n, lapack_int ldA, double* B, lapack_int nrhs)
{
	assert( A );
	assert( m > 0 );
	assert( n > 0 );
	assert( ldA > 0 );
	assert( B );
	assert( nrhs > 0 );

	const lapack_int ldX = std::max(m,n);

	double rc = LapackMatrixRCond(A, m, n, ldA);

	// Make a copy of the input matrix A to avoid changing its content
	lapack_int A_sz = m*n;
	double* AA = new double[A_sz];
	std::copy(A, A+A_sz, AA);

	// Make a copy of the input matrix B to avoid changing its content
	double* X = new double[n*ldX];
	std::copy(B, B+m*nrhs, X);
	std::fill(X+m*nrhs, X+n*ldX, 0);

	lapack_int* jpvt = new lapack_int[n];

	lapack_int rank = 0;

	lapack_int info = 0;
#ifdef FLX_CONFIG_HAVE_LAPACKE
	info = LAPACKE_dgelsy(LAPACK_COL_MAJOR, m, n, nrhs, AA, ldA, X, ldX, jpvt, rc, &rank);
#else
    {
        const lapack_int minmn = std::min(m,n);
        const lapack_int lwork = std::max(1, std::max(minmn+3*n+1, 2*minmn+nrhs));
        double work[lwork];
        dgelsy_(&m, &n, &nrhs, AA, &ldA, X, &ldX, jpvt, &rc, &rank, work, &lwork, &info);
    }
#endif // FLX_CONFIG_HAVE_LAPACKE
	if (info)
	{
		if (AA)
		{
			delete[] AA;
		}
		if (X)
		{
			delete[] X;
		}
		if (jpvt)
		{
			delete[] jpvt;
		}

		std::ostringstream oss;
		oss << "Unable to solve LSQ problem. LAPACK xGELSY returned: " << info;
		throw std::runtime_error(oss.str());
	}
	if (AA)
	{
		AA = 0;
		delete[] AA;
	}
	if (jpvt)
	{
		jpvt = 0;
		delete[] jpvt;
	}

	return X;
}

template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMultiQR(const AMatrixT& A, const BMatrixT& B)
{
    if (A.size() == 0)
    {
        FL_THROW2(std::invalid_argument, "Coefficient matrix is empty");
    }
    if (B.size() == 0)
    {
        FL_THROW2(std::invalid_argument, "Right-hand side matrix is empty");
    }
    if (A.size() != B.size())
    {
        FL_THROW2(std::invalid_argument, "Coefficient matrix and right-hand side matrix are not conformant");
    }

    std::vector< std::vector<ValueT> > X; // The solution matrix X such that: AX=B

    lapack_int A_m = A.size();
    lapack_int A_n = A[0].size();
    lapack_int nrhs = B[0].size();
    lapack_int ldA = A_m;
    lapack_int ldB = std::max(A_m, A_n);

    ValueT* flat_A = 0;
    ValueT* flat_B = 0;

    try
    {
        // Copy A and store it in column-major order
        flat_A = new ValueT[A_m*A_n];
        for (std::size_t j = 0; j < A_n; ++j)
        {
            const std::size_t offs = j*ldA;
            for (std::size_t i = 0; i < A_m; ++i)
            {
                flat_A[offs+i] = A[i][j];
            }
        }
std::cerr << "[LsqQR] A: "; MatrixOutput(std::cerr, A); std::cerr << std::endl;//XXX
std::cerr << "[LsqQR] flat_A: "; VectorOutput(std::cerr, flat_A, A_m*A_n); std::cerr << std::endl;//XXX
        // Copy B and store it in column-major order
        flat_B = new ValueT[A_m*nrhs];
        for (std::size_t j = 0; j < nrhs; ++j)
        {
            const std::size_t offs = j*A_m;
            for (std::size_t i = 0; i < A_m; ++i)
            {
                flat_B[offs+i] = B[i][j];
            }
        }
std::cerr << "[LsqQR] B: "; MatrixOutput(std::cerr, B); std::cerr << std::endl;//XXX
std::cerr << "[LsqQR] flat_B: "; VectorOutput(std::cerr, flat_B, A_m*nrhs); std::cerr << std::endl;//XXX

        double* flat_X = 0;
        flat_X = LapackLsqSolveGELS(flat_A, A_m, A_n, ldA, flat_B, nrhs);
        if (flat_X)
        {
            X.resize(A_n);
            for (lapack_int i = 0; i < A_n; ++i)
            {
                X[i].resize(nrhs);
                for (lapack_int j = 0; j < nrhs; ++j)
                {
                    X[i][j] = flat_B[j*A_n+i];
                }
            }
            delete[] flat_X;
        }
std::cerr << "[LsqQR] X: "; MatrixOutput(std::cerr, X); std::cerr << std::endl;//XXX
    }
    catch (...)
    {
        if (flat_A)
        {
            delete[] flat_A;
        }
        if (flat_B)
        {
            delete[] flat_B;
        }
        throw;
    }

    if (flat_A)
    {
        delete[] flat_A;
    }
    if (flat_B)
    {
        delete[] flat_B;
    }

    return X;
}

template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMultiSVD(const AMatrixT& A, const BMatrixT& B)
{
    if (A.size() == 0)
    {
        FL_THROW2(std::invalid_argument, "Coefficient matrix is empty");
    }
    if (B.size() == 0)
    {
        FL_THROW2(std::invalid_argument, "Right-hand side matrix is empty");
    }
    if (A.size() != B.size())
    {
        FL_THROW2(std::invalid_argument, "Coefficient matrix and right-hand side matrix are not conformant");
    }

    std::vector< std::vector<ValueT> > X; // The solution matrix X such that: AX=B

    lapack_int A_m = A.size();
    lapack_int A_n = A[0].size();
    lapack_int nrhs = B[0].size();
    lapack_int ldA = A_m;

    ValueT* flat_A = 0;
    ValueT* flat_B = 0;

    try
    {
        // Copy A and store it in column-major order
        flat_A = new ValueT[A_m*A_n];
        for (lapack_int j = 0; j < A_n; ++j)
        {
            const lapack_int offs = j*ldA;
            for (lapack_int i = 0; i < A_m; ++i)
            {
                flat_A[offs+i] = A[i][j];
            }
        }
std::cerr << "[LsqSVD] A: "; MatrixOutput(std::cerr, A); std::cerr << std::endl;//XXX
std::cerr << "[LsqSVD] flat_A: "; VectorOutput(std::cerr, flat_A, A_m*A_n); std::cerr << std::endl;//XXX
        // Copy B and store it in column-major order
        flat_B = new ValueT[A_m*nrhs];
        std::fill(flat_B, flat_B+A_m*nrhs, 0);
        for (lapack_int j = 0; j < nrhs; ++j)
        {
            const lapack_int offs = j*A_m;
            for (lapack_int i = 0; i < A_m; ++i)
            {
                flat_B[offs+i] = B[i][j];
            }
        }
std::cerr << "[LsqSVD] B: "; MatrixOutput(std::cerr, B); std::cerr << std::endl;//XXX
std::cerr << "[LsqSVD] flat_B: "; VectorOutput(std::cerr, flat_B, A_m*nrhs); std::cerr << std::endl;//XXX

        double* flat_X = 0;
        flat_X = LapackLsqSolveGELSD(flat_A, A_m, A_n, ldA, flat_B, nrhs);
        //flat_X = LapackLsqSolveGELSS(flat_A, A_m, A_n, ldA, flat_B, nrhs);
        //flat_X = LapackLsqSolveGELSY(flat_A, A_m, A_n, ldA, flat_B, nrhs);
        if (flat_X)
        {
            X.resize(A_n);
            for (lapack_int i = 0; i < A_n; ++i)
            {
                X[i].resize(nrhs);
                for (lapack_int j = 0; j < nrhs; ++j)
                {
                    X[i][j] = flat_X[j*A_n+i];
                }
            }
            delete[] flat_X;
        }
std::cerr << "[LsqQR] X: "; MatrixOutput(std::cerr, X); std::cerr << std::endl;//XXX
    }
    catch (...)
    {
        if (flat_A)
        {
            delete[] flat_A;
        }
        if (flat_B)
        {
            delete[] flat_B;
        }
        throw;
    }

    if (flat_A)
    {
        delete[] flat_A;
    }
    if (flat_B)
    {
        delete[] flat_B;
    }

    return X;
}

template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMulti(const AMatrixT& A, const BMatrixT& B)
{
    // NOTE: QR implementation is faster but more sensible to ill-conditioned problems.
    //       SVD implementation is slower but more robust.

std::vector< std::vector<ValueT> > X;
#if 0
    X = LsqSolveMultiQR<ValueT>(A, B);
#else
    X = LsqSolveMultiSVD<ValueT>(A, B);
#endif
return X;
}

template <typename ValueT, typename MatrixT, typename VectorT>
std::vector<ValueT> LsqSolve(const MatrixT& A, const VectorT& b)
{
    std::vector< std::vector<ValueT> > B(b.size());
    for (std::size_t i = 0,
                     ni = b.size();
         i < ni;
         ++i)
    {
        B[i].push_back(b[i]);
    }

    std::vector< std::vector<ValueT> > X;
    X = LsqSolveMulti<ValueT>(A, B);

    std::vector<ValueT> x(X.size());
    for (std::size_t i = 0,
                     ni = X.size();
         i < ni;
         ++i)
    {
        x[i] = X[i][0];
    }

    return x;
}

#else // FLX_CONFIG_HAVE_LAPACK

////////////////////////////////////////////////////////
// SVDDecomposition


/**
 * Singular Value Decomposition
 *
 * In linear algebra, the singular value decomposition (SVD) is a factorization of a real or complex matrix
 *
 * Suppose \f$\mathbf{A}\f$ is a \f$m \times n\f$ real or complex matrix.
 * Then there exists a factorization of the form
 * \f[
 *  \mathbf{A} = \mathbf{U} \mathbf{W} \mathbf{V}^*
 * \f]
 * where \f$\mathbf{U}\f$ is an \f$m \times m\f$ unitary matrix (orthogonal
 * matrix if \f$\mathbf{A}\f$ is a real matrix), \f$\mathbf{W}\f$ is a
 * \f$m \times n\f$ diagonal matrix with non-negative real numbers on the
 * diagonal, and the \f$n \times n\f$ unitary matrix \f$\mathbf{V}^∗\f$
 * denotes the conjugate transpose of the \f$n \times n\f$ unitary matrix
 * \f$\mathbf{V}\f$.
 * Such a factorization is called a singular value decomposition of \f$\mathbf{A}\f$.
 *
 * The diagonal entries of \f$\mathbf{W}\f$ are known as the singular values of
 * \f$\mathbf{A}\f$.
 * A common convention is to list the singular values in descending order.
 * In this case, the diagonal matrix \f$\mathbf{W}\f$ is uniquely determined by
 * \f$\mathbf{A}\f$ (though the matrices \f$\mathbf{U}\f$ and \f$\mathbf{V}\f$
 * are not).
 * 
 * This implementation is mostly taken from (Press et al., 2007).
 *
 * References
 * -# Wikipedia, "Singular value decomposition," Available online: https://en.wikipedia.org/wiki/Singular_value_decomposition.
 * -# W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery, "Numerical Recipies: The Art of Scientific Computing," 3rd Edition, Cambridge University Press, 2007.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <class RealT>
class SVDDecomposition
{
public:
    /// Default constructor
    SVDDecomposition()
    : m_(0),
      n_(0)
    {
    }

    /// Performs the SVD computation of the given matrix
    template <typename MatrixT>
    SVDDecomposition(const MatrixT& A)
    {
        decompose(A);
    }

    /// Performs the SVD computation of the given matrix
    template <typename MatrixT>
    void decompose(const MatrixT& A)
    {
        m_ = A.size();
        n_ = (A.size() > 0) ? A[0].size() : 0;

        MatrixCopy(A, u_);
        MatrixCopy(MatrixZero<RealT>(n_, n_), v_);
        VectorCopy(VectorZero<RealT>(n_), w_);

        decompose();
        reorder();
    }

    /**
     * Solves the system \f$\mathbf{A}\mathbf{x}=\mathbf{b}\f$ for a vector
     * \f$\mathbf{x}\f$ using the pseudoinverse of \f$\mathbf{A}\f$ as otained
     * by SVD.
     *
     * If positive, thresh is the threshold value below which singular values are considered as zero.
     * If thresh is negative, a default based on expected roundoff error is used.
     */
    template <typename VectorT>
    std::vector<RealT> solve(const VectorT& b, RealT thresh = -1) const;

    /**
     * Solves the system \f$\mathbf{A}\mathbf{X}=\mathbf{B}\f$ for a matrix
     * \f$\mathbf{X}\f$ using the pseudoinverse of \f$\mathbf{A}\f$ as otained
     * by SVD.
     *
     * If positive, thresh is the threshold value below which singular values
     * are considered as zero.
     * If thresh is negative, a default based on expected roundoff error is used.
     */
    template <typename MatrixT>
    std::vector< std::vector<RealT> > solveMulti(const MatrixT& b, RealT thresh = -1) const;

    /**
     * Return the rank of A, after zeroing any singular values smaller than thresh.
     *
     * If positive, thresh is the threshold value below which singular values
     * are considered as zero.
     * If thresh is negative, a default based on expected roundoff error is used.
     */
	std::size_t rank(RealT thresh = -1) const;

    /**
     * Return the nullity of A, after zeroing any singular values smaller than thresh.
     *
     * If positive, thresh is the threshold value below which singular values
     * are considered as zero.
     * If thresh is negative, a default based on expected roundoff error is used.
     */
	std::size_t nullity(RealT thresh = -1) const;

    /**
     * Give an orthonormal basis for the range of \f$\mathbf{A}\f$ as the
     * columns of a returned matrix.
     *
     * If positive, thresh is the threshold value below which singular values
     * are considered as zero.
     * If thresh is negative, a default based on expected roundoff error is used.
     */
	std::vector< std::vector<RealT> > range(RealT thresh = -1) const;

    /**
     * Give an orthonormal basis for the nullspace of \f$\mathbf{A}\f$ as the
     * columns of a returned matrix.
     *
     * If positive, thresh is the threshold value below which singular values
     * are considered as zero.
     * If thresh is negative, a default based on expected roundoff error is used.
     */
	std::vector< std::vector<RealT> > nullspace(RealT thresh = -1) const;


private:
    /// Performs the SVD computation
	void decompose();

	void reorder();

	static RealT Pythag(RealT a, RealT b);

    static RealT Sign(RealT a, RealT b);

    RealT getDefaultThreshold() const;

    RealT getInverseConditionNumber() const;


private:
    std::size_t m_;
    std::size_t n_;
    std::vector< std::vector<RealT> > u_; ///< The matrix U
    std::vector< std::vector<RealT> > v_; ///< The matrix V
    std::vector<RealT> w_; ///< The diagonal matrix W
}; // SVDDecomposition

template <typename RealT>
template <typename VectorT>
std::vector<RealT> SVDDecomposition<RealT>::solve(const VectorT& b, RealT thresh) const
{
	if (b.size() != m_)
    {
        FL_THROW2(std::invalid_argument, "Wrong dimension for the coefficient vector");
    }

    const RealT tsh = (thresh >= 0) ? thresh : this->getDefaultThreshold();

    std::vector<RealT> x(n_);

	std::vector<RealT> tmp(n_);
	for (std::size_t j = 0; j < n_; ++j)
    {
		RealT s = 0;
		if (w_[j] > tsh)
        {
			for (std::size_t i = 0; i < m_; ++i)
            {
                s += u_[i][j]*b[i];
            }
			s /= w_[j];
		}
		tmp[j] = s;
	}
	for (std::size_t j = 0; j < n_; ++j)
    {
		RealT s = 0;
		for (std::size_t jj = 0; jj < n_; ++jj)
        {
            s += v_[j][jj]*tmp[jj];
        }
		x[j] = s;
	}

    return x;
}

template <typename RealT>
template <typename MatrixT>
std::vector< std::vector<RealT> > SVDDecomposition<RealT>::solveMulti(const MatrixT& B, RealT thresh) const
{
	if (B.size() != m_)
    {
        FL_THROW2(std::invalid_argument, "Wrong dimension for the coefficient matrix");
    }

    const std::size_t p = B[0].size();

    std::vector< std::vector<RealT> > X;
    MatrixCopy(MatrixZero<RealT>(n_, p), X);

	for (std::size_t j = 0; j < p;  ++j)
    {
        std::vector<RealT> Bcol(m_);

		for (std::size_t i = 0; i < m_; ++i)
        {
            Bcol[i] = B[i][j];
        }

	    const std::vector<RealT> x = this->solve(Bcol, thresh);

		for (std::size_t i = 0; i < n_; ++i)
        {
            X[i][j] = x[i];
        }
	}

    return X;
}

template <typename RealT>
std::size_t SVDDecomposition<RealT>::rank(RealT thresh) const
{
    const RealT tsh = (thresh >= 0) ? thresh : this->getDefaultThreshold();

	std::size_t nr = 0;
	for (std::size_t j = 0; j < n_; ++j)
    {
        if (w_[j] > tsh)
        {
            nr++;
        }
    }

	return nr;
}

template <typename RealT>
std::size_t SVDDecomposition<RealT>::nullity(RealT thresh) const
{
    const RealT tsh = (thresh >= 0) ? thresh : this->getDefaultThreshold();

	std::size_t nn = 0;
	for (std::size_t j = 0; j < n_; ++j)
    {
        if (w_[j] <= tsh)
        {
            ++nn;
        }
    }

	return nn;
}

template <typename RealT>
std::vector< std::vector<RealT> > SVDDecomposition<RealT>::range(RealT thresh) const
{
    const RealT tsh = (thresh >= 0) ? thresh : this->getDefaultThreshold();

	std::size_t nr = 0;
	std::vector< std::vector<RealT> > rnge(m_, this->rank(thresh));
	for (std::size_t j = 0; j < n_; ++j)
    {
		if (w_[j] > tsh) {
			for (std::size_t i = 0; i < m_; ++i)
            {
                rnge[i][nr] = u_[i][j];
            }
			++nr;
		}
	}
	return rnge;
}

template <typename RealT>
std::vector< std::vector<RealT> > SVDDecomposition<RealT>::nullspace(RealT thresh) const
{
    const RealT tsh = (thresh >= 0) ? thresh : this->getDefaultThreshold();

	std::vector< std::vector<RealT> > nullsp(n_, this->nullity(thresh));

	std::size_t nn = 0;
    for (std::size_t j = 0; j < n_; ++j)
    {
        if (w_[j] <= tsh)
        {
            for (std::size_t jj = 0; jj < n_; ++jj)
            {
                nullsp[jj][nn] = v_[jj][j];
            }
            ++nn;
        }
    }

    return nullsp;
}

template <typename RealT>
void SVDDecomposition<RealT>::decompose()
{
    const RealT eps = std::numeric_limits<RealT>::epsilon();

	bool flag;
	//int i,its,j,jj,k,l,nm;
	//RealT anorm,c,f,g,h,s,scale,x,y,z;
	std::vector<RealT> rv1(n_);
	RealT g = 0;
    RealT scale = 0;
    RealT anorm = 0;
    int l = 0;
    int nm = 0;
	for (int i = 0; i < n_; ++i)
    {
        RealT s = 0;

		l = i+2;
		rv1[i] = scale*g;
		g = scale = 0;

		if (i < m_)
        {
			for (int k = i; k < m_; ++k)
            {
                scale += std::abs(u_[k][i]);
            }
			if (scale != 0.0)
            {
				for (int k = i; k < m_; ++k)
                {
					u_[k][i] /= scale;
					s += Sqr(u_[k][i]);
				}
				RealT f = u_[i][i];
				g = -Sign(std::sqrt(s), f);
				RealT h = f*g - s;
				u_[i][i] = f-g;
				for (int j = l-1; j < n_; ++j)
                {
                    s = 0;
					for (int k = i; k < m_; ++k)
                    {
                        s += u_[k][i]*u_[k][j];
                    }
					f = s/h;
					for (int k = i; k < m_; ++k)
                    {
                        u_[k][j] += f*u_[k][i];
                    }
				}
				for (int k = i; k < m_; ++k)
                {
                    u_[k][i] *= scale;
                }
			}
		}
		w_[i] = scale*g;
		g = s = scale = 0;
		if ((i+1) <= m_ && (i+1) != n_)
        {
			for (int k = l-1; k < n_; ++k)
            {
                scale += std::abs(u_[i][k]);
            }
			if (scale != 0.0)
            {
				for (int k = l-1; k < n_; ++k)
                {
					u_[i][k] /= scale;
					s += Sqr(u_[i][k]);
				}
				RealT f = u_[i][l-1];
				g = -Sign(std::sqrt(s), f);
				RealT h = f*g - s;
				u_[i][l-1] = f-g;
				for (int k = l-1; k < n_; ++k)
                {
                    rv1[k] = u_[i][k]/h;
                }
				for (int j = l-1; j < m_; ++j)
                {
                    s = 0;
					for (int k = l-1; k < n_; ++k)
                    {
                        s += u_[j][k]*u_[i][k];
                    }
					for (int k = l-1; k < n_; ++k)
                    {
                        u_[j][k] += s*rv1[k];
                    }
				}
				for (int k = l-1; k < n_; ++k)
                {
                    u_[i][k] *= scale;
                }
			}
		}
		anorm = std::max(anorm,(std::abs(w_[i])+std::abs(rv1[i])));
	}
	for (int i = n_-1; i >= 0; --i)
    {
		if (i < (n_-1))
        {
			if (g != 0.0)
            {
				for (int j = l; j < n_; ++j)
                {
					v_[j][i]=(u_[i][j]/u_[i][l])/g;
                }
				for (int j = l; j < n_; ++j)
                {
                    RealT s = 0;
					for (int k = l; k < n_; ++k)
                    {
                        s += u_[i][k]*v_[k][j];
                    }
					for (int k = l; k < n_; ++k)
                    {
                        v_[k][j] += s*v_[k][i];
                    }
				}
			}
			for (int j = l; j < n_; ++j)
            {
                v_[i][j] = v_[j][i] = 0.0;
            }
		}
		v_[i][i]=1.0;
		g = rv1[i];
		l = i;
	}
	for (int i = std::min(m_, n_)-1; i >= 0; --i)
    {
		l = i+1;
		g = w_[i];
		for (int j = l; j < n_; ++j)
        {
            u_[i][j]=0.0;
        }
		if (g != 0.0)
        {
			g = 1.0/g;
			for (int j = l; j < n_; ++j)
            {
                RealT s = 0;
				for (int k = l; k < m_; ++k)
                {
                    s += u_[k][i]*u_[k][j];
                }
				RealT f = (s/u_[i][i])*g;
				for (int k = i; k < m_; ++k)
                {
                    u_[k][j] += f*u_[k][i];
                }
			}
			for (int j = i; j < m_; ++j)
            {
                u_[j][i] *= g;
            }
		}
        else
        {
            for (int j = i; j < m_; ++j)
            {
                u_[j][i]=0.0;
            }
        }
		++u_[i][i];
	}
	for (int k = n_-1; k >= 0; --k)
    {
		for (int its = 0; its < 30; ++its)
        {
			flag = true;
			for (l = k; l >= 0; --l)
            {
				nm = l-1;
				if (l == 0 || std::abs(rv1[l]) <= (eps*anorm))
                {
					flag = false;
					break;
				}
				if (std::abs(w_[nm]) <= (eps*anorm))
                {
                    break;
                }
			}
			if (flag)
            {
				RealT c = 0.0;
				RealT s = 1.0;
				for (int i = l; i < (k+1); ++i)
                {
					RealT f = s*rv1[i];
					rv1[i] = c*rv1[i];
					if (std::abs(f) <= (eps*anorm))
                    {
                        break;
                    }
					g = w_[i];
					RealT h = Pythag(f,g);
					w_[i] = h;
					h = 1.0/h;
					c = g*h;
					s = -f*h;
					for (int j = 0; j < m_; ++j)
                    {
						RealT y = u_[j][nm];
						RealT z = u_[j][i];
						u_[j][nm] = y*c + z*s;
						u_[j][i] = z*c - y*s;
					}
				}
			}
			RealT z = w_[k];
			if (l == k)
            {
				if (z < 0)
                {
					w_[k] = -z;
					for (int j = 0; j < n_; ++j)
                    {
                        v_[j][k] = -v_[j][k];
                    }
				}
				break;
			}
			if (its == 29)
            {
                throw("no convergence in 30 svdcmp iterations");
            }
			RealT x = w_[l];
			nm = k-1;
			RealT y = w_[nm];
			g = rv1[nm];
			RealT h = rv1[k];
			RealT f = ((y-z)*(y+z) + (g-h)*(g+h))/(2.0*h*y);
			g = Pythag(f, 1.0);
			f = ((x-z)*(x+z) + h*((y/(f+Sign(g, f)))-h))/x;
            RealT c = 1;
            RealT s = 1;
			for (int j = l; j <= nm; ++j)
            {
				int i = j+1;
				g = rv1[i];
				y = w_[i];
				h = s*g;
				g = c*g;
				RealT z = Pythag(f, h);
				rv1[j] = z;
				c = f/z;
				s = h/z;
				f = x*c + g*s;
				g = g*c - x*s;
				h = y*s;
				y *= c;
				for (int jj = 0; jj < n_; ++jj)
                {
					x = v_[jj][j];
					z = v_[jj][i];
					v_[jj][j] = x*c + z*s;
					v_[jj][i] = z*c - x*s;
				}
				z = Pythag(f, h);
				w_[j] = z;
				if (z)
                {
					z = 1.0/z;
					c = f*z;
					s = h*z;
				}
				f = c*g + s*y;
				x = c*y - s*g;
				for (int jj = 0; jj < m_; ++jj)
                {
					y = u_[jj][j];
					z = u_[jj][i];
					u_[jj][j] = y*c + z*s;
					u_[jj][i] = z*c - y*s;
				}
			}
			rv1[l] = 0;
			rv1[k] = f;
			w_[k] = x;
		}
	}
}

template <typename RealT>
void SVDDecomposition<RealT>::reorder()
{
    int inc = 1;
	do
    {
        inc *= 3;
        ++inc;
    }
    while (inc <= n_);

	std::vector<RealT> su(m_);
    std::vector<RealT> sv(n_);
	do
    {
		inc /= 3;
		for (int i = inc; i < n_; ++i)
        {
			RealT sw = w_[i];
			for (int k = 0; k < m_; ++k)
            {
                su[k] = u_[k][i];
            }
			for (int k = 0; k < n_; ++k)
            {
                sv[k] = v_[k][i];
            }
			int j = i;
			while (w_[j-inc] < sw)
            {
				w_[j] = w_[j-inc];
				for (int k = 0; k < m_; ++k)
                {
                    u_[k][j] = u_[k][j-inc];
                }
				for (int k = 0; k < n_; ++k)
                {
                    v_[k][j] = v_[k][j-inc];
                }
				j -= inc;
				if (j < inc)
                {
                    break;
                }
			}
			w_[j] = sw;
			for (int k = 0; k < m_; ++k)
            {
                u_[k][j] = su[k];
            }
			for (int k = 0; k < n_; ++k)
            {
                v_[k][j] = sv[k];
            }
		}
	}
    while (inc > 1);

	for (int k = 0; k < n_; ++k)
    {
		RealT s = 0;
		for (int i = 0; i < m_; ++i)
        {
            if (u_[i][k] < 0.)
            {
                ++s;
            }
        }
		for (int j = 0; j < n_; ++j)
        {
            if (v_[j][k] < 0.)
            {
                ++s;
            }
        }
		if (s > (m_+n_)/2)
        {
			for (int i = 0; i < m_; ++i)
            {
                u_[i][k] = -u_[i][k];
            }
			for (int j = 0; j < n_; ++j)
            {
                v_[j][k] = -v_[j][k];
            }
		}
	}
}

template <typename RealT>
RealT SVDDecomposition<RealT>::getInverseConditionNumber() const
{
    return (w_[0] <= 0. || w_[n_-1] <= 0.) ? 0. : w_[n_-1]/w_[0];
}

template <typename RealT>
RealT SVDDecomposition<RealT>::getDefaultThreshold() const
{
    const RealT eps = std::numeric_limits<RealT>::epsilon();

    return 0.5*std::sqrt(m_+n_+1.0)*w_[0]*eps;
}

template <typename RealT>
RealT SVDDecomposition<RealT>::Pythag(RealT a, RealT b)
{
	RealT absa = std::abs(a);
    RealT absb = std::abs(b);

	return (absa > absb) ?
           (absa*std::sqrt(1.0+Sqr(absb/absa)))
           : ((absb == 0.0)
              ? 0.0
              : absb*std::sqrt(1.0+Sqr(absa/absb)));
}

template <typename RealT>
RealT SVDDecomposition<RealT>::Sign(RealT a, RealT b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

////////////////////////////////////////////////////////


template <typename ValueT, typename MatrixT, typename VectorT>
std::vector<ValueT> LsqSolve(const MatrixT& A, const VectorT& b)
{
    SVDDecomposition<ValueT> svd(A);

    return svd.solve(b);
}

template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMulti(const AMatrixT& A, const BMatrixT& B)
{
    SVDDecomposition<ValueT> svd(A);

    return svd.solveMulti(B);
}

#endif // FLX_CONFIG_HAVE_LAPACK

}} // Namespace fl::detail


#endif // FL_DETAIL_LSQ_H

/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */
