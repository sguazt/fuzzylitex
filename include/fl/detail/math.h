/**
 * \file fl/detail/math.h
 *
 * \brief Mathematical utilities
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

#ifndef FL_DETAIL_MATH_H
#define FL_DETAIL_MATH_H


#include <cmath>
#include <cstddef>
#include <fl/macro.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>


namespace fl { namespace detail {

////////////////////////////////////////////////////////////////////////////////
/// Declarations
////////////////////////////////////////////////////////////////////////////////


/// Returns the sum (or the difference, if \a minus is true) of the values in the range [\a first, \a last)
template <typename T, typename IterT>
T AlgebraicSum(IterT first, IterT last, bool minus);

/// Returns the difference of the values in the range [\a first, \a last)
template <typename T, typename IterT>
T Diff(IterT first, IterT last);

/// Returns evenly \a spaced values over the given interval [\a start, \a stop]. The end-point of the interval can be optionally excluded.
template <typename ValueT>
std::vector<ValueT> LinSpace(ValueT start, ValueT stop, std::size_t num, bool endPoint = true);

template <typename ValueT, typename MatrixT, typename VectorT>
std::vector<ValueT> LsqSolve(const MatrixT& A, const VectorT& b);

template <typename ValueT, typename AMatrixT, typename BMatrixT>
std::vector< std::vector<ValueT> > LsqSolveMulti(const AMatrixT& A, const BMatrixT& B);

/// Computes A+B, where \a A and \a B are nxm matrices
template <typename MatrixT>
MatrixT MatrixAlgebraicSum(const MatrixT& A, const MatrixT& B, bool minus);

/// Copies matrix \a in into matrix \a out
template <typename InMatrixT, typename OutMatrixT>
OutMatrixT& MatrixCopy(const InMatrixT& in, OutMatrixT& out);

/// Computes A-B, where A and B are nxm matrices
template <typename MatrixT>
MatrixT MatrixDiff(const MatrixT& A, const MatrixT& B);

/// Computes the transpose A' of A, where \a A is a nxm matrix
template <typename MatrixT>
MatrixT MatrixIdentity(std::size_t nr, std::size_t nc);

template <typename MatrixT>
typename MatrixT::value_type MatrixMax(const MatrixT& A);

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMaxColumn(const MatrixT& A);

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMaxRow(const MatrixT& A);

template <typename MatrixT>
typename MatrixT::value_type MatrixMin(const MatrixT& A);

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMinColumn(const MatrixT& A);

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMinRow(const MatrixT& A);

/// Outputs the given matrix \a A to the output stream \a os
template <typename CharT, typename CharTraitsT, typename MatrixT>
void MatrixOutput(std::basic_ostream<CharT,CharTraitsT>& os, const MatrixT& A);

/// Outputs the given matrix \a A to the output stream \a os
template <typename CharT, typename CharTraitsT, typename MatrixT>
void MatrixOutput(std::basic_ostream<CharT,CharTraitsT>& os, const MatrixT& A, std::size_t nrows, std::size_t ncols);

/// Computes A*B, where \a A is a nxp matrix and \a B is a pxm matrix
template <typename MatrixT>
MatrixT MatrixProduct(const MatrixT& A, const MatrixT& B);

/// Computes A*c, where \a A is a nxm matrix and \a c is a scalar value
template <typename MatrixT, typename ValueT>
MatrixT MatrixScalarProduct(const MatrixT& A, ValueT c);

/// Computes A+B, where \a A and \a B are nxm matrices
template <typename MatrixT>
MatrixT MatrixSum(const MatrixT& A, const MatrixT& B);

/// Computes the transpose A' of A, where \a A is a nxm matrix
template <typename MatrixT>
MatrixT MatrixTranspose(const MatrixT& A);

/// Computes A*v, where \a A is a nxp matrix and \a v is a px1 vector
template <typename MatrixT, typename VectorT>
VectorT MatrixVectorProduct(const MatrixT& A, const VectorT& v);

template <typename T>
std::vector< std::vector<T> > MatrixZero(std::size_t nr, std::size_t nc);

/// Returns the square of the given parameter \a x
template <typename T>
T Sqr(T x);

/// Returns the sum of the values in the range [\a first, \a last)
template <typename T, typename IterT>
T Sum(IterT first, IterT last);

/// Computes u+v (or u-v, if \a minus is true), where \a u and \a v are nx1 vectors
template <typename VectorT>
VectorT VectorAlgebraicSum(const VectorT& u, const VectorT& v, bool minus);

/// Copies vector \a in into vector \a out
template <typename InVectorT, typename OutVectorT>
OutVectorT& VectorCopy(const InVectorT& in, OutVectorT& out);

/// Computes u-v, where \a u and \a v are nx1 vectors
template <typename VectorT>
VectorT VectorDiff(const VectorT& u, const VectorT& v);

/// Computes the inner product u'*v, where \a u and \a v are nx1 vectors, and u' is the transpose of u
template <typename VectorT>
typename VectorT::value_type VectorInnerProduct(const VectorT& u, const VectorT& v);

/// Computes v'*A, where \a v is a px1 vector, \a A is a pxn matrix, and v' is the transpose of v
template <typename VectorT, typename MatrixT>
VectorT VectorMatrixProduct(const VectorT& v, const MatrixT& A);

template <typename VectorT>
typename VectorT::value_type VectorMax(const VectorT& v);

template <typename VectorT>
typename VectorT::value_type VectorMin(const VectorT& v);

template <typename VectorT>
typename VectorT::value_type VectorNorm(VectorT& v, unsigned int p = 2);

/// Computes the outer product u*v', where \a u and \a v are nx1 vectors, and v' is the transpose of v
template <typename MatrixT, typename VectorT>
MatrixT VectorOuterProduct(const VectorT& u, const VectorT& v);

/// Outputs the given vector \a v to the output stream \a os
template <typename CharT, typename CharTraitsT, typename VectorT>
void VectorOutput(std::basic_ostream<CharT,CharTraitsT>& os, const VectorT& v, std::size_t n);

/// Outputs the given vector \a v to the output stream \a os
template <typename CharT, typename CharTraitsT, typename VectorT>
void VectorOutput(std::basic_ostream<CharT,CharTraitsT>& os, const VectorT& v);

/// Computes v*c, where \a v is a nx1 vector and \a c is a scalar value
template <typename VectorT, typename ValueT>
VectorT VectorScalarProduct(const VectorT& v, ValueT c);

/// Computes u+v, where \a u and \a v are nx1 vectors
template <typename VectorT>
VectorT VectorSum(const VectorT& u, const VectorT& v);

template <typename T>
std::vector<T> VectorZero(std::size_t n);


////////////////////////////////////////////////////////////////////////////////
/// Definitions
////////////////////////////////////////////////////////////////////////////////


template <typename T>
T Sqr(T x)
{
    return x*x;
}

template <typename T, typename IterT>
T AlgebraicSum(IterT first, IterT last, bool minus)
{
    T sum = 0;
    while (first != last)
    {
        if (minus)
        {
            sum -= *first;
        }
        else
        {
            sum += *first;
        }
        ++first;
    }
    return sum;
}

template <typename T, typename IterT>
T Sum(IterT first, IterT last)
{
    return AlgebraicSum<T>(first, last, false);
}

template <typename T, typename IterT>
T Diff(IterT first, IterT last)
{
    return AlgebraicSum<T>(first, last, true);
}

template <typename ValueT>
std::vector<ValueT> LinSpace(ValueT start, ValueT stop, std::size_t num, bool endPoint)
{
    std::vector<ValueT> res(num);

    if (endPoint)
    {
        if (num == 1)
        {
            res[0] = start;
        }
        else
        {
            const ValueT step = (stop-start)/(num-1);
            for (std::size_t i = 0; i < num; ++i)
            {
                res[i] = i*step+start;
            }
            res[num-1] = stop;
        }
    }
    else
    {
        const ValueT step = (stop-start)/num;
        for (std::size_t i = 0; i < num; ++i)
        {
            res[i] = i*step+start;
        }
    }

    return res;
}

template <typename CharT, typename CharTraitsT, typename VectorT>
void VectorOutput(std::basic_ostream<CharT,CharTraitsT>& os, const VectorT& v)
{
/*
    os << "[";
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (i > 0)
        {
            os << " ";
        }
        os << v[i];
    }
    os << "]";
*/
    VectorOutput(os, v, v.size());
}

template <typename CharT, typename CharTraitsT, typename VectorT>
void VectorOutput(std::basic_ostream<CharT,CharTraitsT>& os, const VectorT& v, std::size_t n)
{
    os << "[";
    for (std::size_t i = 0; i < n; ++i)
    {
        if (i > 0)
        {
            os << " ";
        }
        os << v[i];
    }
    os << "]";
}

template <typename CharT, typename CharTraitsT, typename MatrixT>
void MatrixOutput(std::basic_ostream<CharT,CharTraitsT>& os, const MatrixT& A)
{
/*
    os << "[";
    for (std::size_t i = 0; i < A.size(); ++i)
    {
        if (i > 0)
        {
            os << "; ";
        }
        for (std::size_t j = 0; j < A[i].size(); ++j)
        {
            if (j > 0)
            {
                os << " ";
            }
            os << A[i][j];
        }
    }
    os << "]";
*/
    MatrixOutput(os, A, A.size(), A.size() > 0 ? A[0].size() : 0);
}

template <typename CharT, typename CharTraitsT, typename MatrixT>
void MatrixOutput(std::basic_ostream<CharT,CharTraitsT>& os, const MatrixT& A, std::size_t nrows, std::size_t ncols)
{
    os << "[";
    for (std::size_t i = 0; i < nrows; ++i)
    {
        if (i > 0)
        {
            os << "; ";
        }
        for (std::size_t j = 0; j < ncols; ++j)
        {
            if (j > 0)
            {
                os << " ";
            }
            os << A[i][j];
        }
    }
    os << "]";
}

template <typename VectorT, typename ValueT>
VectorT VectorScalarProduct(const VectorT& v, ValueT c)
{
    const std::size_t n = v.size();

    VectorT res(n, 0);

    for (std::size_t i = 0; i < n; ++i)
    {
        res[i] = c*v[i];
    }

    return res;
}

template <typename MatrixT, typename ValueT>
MatrixT MatrixScalarProduct(const MatrixT& A, ValueT c)
{
    const std::size_t nra = A.size();
    const std::size_t nca = A[0].size();

    MatrixT res(nra);

    for (std::size_t i = 0; i < nra; ++i)
    {
        res[i].resize(nca, 0);
        for (std::size_t j = 0; j < nca; ++j)
        {
            res[i][j] = c*A[i][j];
        }
    }

    return res;
}

template <typename VectorT>
VectorT VectorAlgebraicSum(const VectorT& u, const VectorT& v, bool minus)
{
    const std::size_t nu = u.size();
    const std::size_t nv = v.size();

    if (nu != nv)
    {
        FL_THROW2(std::invalid_argument, "Vector dimensions not conformant for vector sum");
    }

    VectorT res(nu, 0);

    for (std::size_t i = 0; i < nu; ++i)
    {
        res[i] = minus ? (u[i]-v[i]) : (u[i]+v[i]);
    }

    return res;
}

template <typename VectorT>
VectorT VectorSum(const VectorT& u, const VectorT& v)
{
    return VectorAlgebraicSum(u, v, false);
}

template <typename VectorT>
VectorT VectorDiff(const VectorT& u, const VectorT& v)
{
    return VectorAlgebraicSum(u, v, true);
}

template <typename MatrixT>
MatrixT MatrixTranspose(const MatrixT& A)
{
    const std::size_t nr = A.size();
    const std::size_t nc = A[0].size();

    MatrixT res(nc);

    for (std::size_t i = 0; i < nr; ++i)
    {
        res[i].resize(nr, 0);
        for (std::size_t j = 0; j < nc; ++j)
        {
            res[j][i] = A[i][j];
        }
    }

    return res;
}

template <typename MatrixT>
MatrixT MatrixIdentity(std::size_t nr, std::size_t nc)
{
    MatrixT res(nr);

    for (std::size_t i = 0; i < nr; ++i)
    {
        res[i].resize(nc, 0);
        res[i][i] = 1;
    }

    return res;
}

template <typename MatrixT>
MatrixT MatrixAlgebraicSum(const MatrixT& A, const MatrixT& B, bool minus)
{
    const std::size_t nra = A.size();
    const std::size_t nca = A[0].size();
    const std::size_t nrb = B.size();
    const std::size_t ncb = B[0].size();

    if (nra != nrb || nca != ncb)
    {
        FL_THROW2(std::invalid_argument, "Matrix dimensions not conformant for matrix sum");
    }

    MatrixT res(nra);

    for (std::size_t i = 0; i < nra; ++i)
    {
        res[i].resize(nca, 0);
        for (std::size_t j = 0; j < nca; ++j)
        {
            res[i][j] = minus ? (A[i][j]-B[i][j]) : (A[i][j]+B[i][j]);
        }
    }

    return res;
}

template <typename MatrixT>
MatrixT MatrixSum(const MatrixT& A, const MatrixT& B)
{
    return MatrixAlgebraicSum(A, B, false);
}

template <typename MatrixT>
MatrixT MatrixDiff(const MatrixT& A, const MatrixT& B)
{
    return MatrixAlgebraicSum(A, B, true);
}

template <typename MatrixT>
MatrixT MatrixProduct(const MatrixT& A, const MatrixT& B)
{
    const std::size_t nra = A.size();
    const std::size_t nca = A[0].size();
    const std::size_t nrb = B.size();
    const std::size_t ncb = B[0].size();

    if (nca != nrb)
    {
        FL_THROW2(std::invalid_argument, "Matrix dimensions not conformant for matrix product");
    }

    MatrixT res(nra);

    for (std::size_t i = 0; i < nra; ++i)
    {
        res[i].resize(ncb, 0);
        for (std::size_t k = 0; k < ncb; ++k)
        {
            for (std::size_t j = 0; j < nca; ++j)
            {
                res[i][k] += A[i][j]*B[j][k];
            }
        }
    }

    return res;
}

template <typename VectorT>
typename VectorT::value_type VectorInnerProduct(const VectorT& u, const VectorT& v)
{
    const std::size_t nu = u.size();
    const std::size_t nv = v.size();

    if (nu != nv)
    {
        FL_THROW2(std::invalid_argument, "Vector dimensions not conformant for vector inner-product");
    }

    typename VectorT::value_type res = 0;

    for (std::size_t i = 0; i < nu; ++i)
    {
        res += u[i]*v[i];
    }

    return res;
}
 
template <typename MatrixT, typename VectorT>
MatrixT VectorOuterProduct(const VectorT& u, const VectorT& v)
{
    const std::size_t nu = u.size();
    const std::size_t nv = v.size();

    MatrixT res(nu);

    for (std::size_t i = 0; i < nu; ++i)
    {
        res[i].resize(nv, 0);
        for (std::size_t j = 0; j < nv; ++j)
        {
            res[i][j] = u[i]*v[j];
        }
    }

    return res;
}
 
template <typename MatrixT, typename VectorT>
VectorT MatrixVectorProduct(const MatrixT& A, const VectorT& v)
{
    const std::size_t nra = A.size();
    const std::size_t nca = A[0].size();
    const std::size_t nv = v.size();

    if (nca != nv)
    {
        FL_THROW2(std::invalid_argument, "Matrix/vector dimensions not conformant for matrix-vector product");
    }

    VectorT res(nra, 0);

    for (std::size_t i = 0; i < nra; ++i)
    {
        for (std::size_t j = 0; j < nca; ++j)
        {
            res[i] += A[i][j]*v[j];
        }
    }

    return res;
}

template <typename VectorT, typename MatrixT>
VectorT VectorMatrixProduct(const VectorT& v, const MatrixT& A)
{
    const std::size_t nra = A.size();
    const std::size_t nca = A[0].size();
    const std::size_t nv = v.size();

    if (nra != nv)
    {
        FL_THROW2(std::invalid_argument, "Matrix/vector dimensions not conformant for vector-matrix product");
    }

    VectorT res(nca, 0);

    for (std::size_t j = 0; j < nca; ++j)
    {
        for (std::size_t i = 0; i < nra; ++i)
        {
            res[j] += A[i][j]*v[i];
        }
    }

    return res;
}

template <typename VectorT>
typename VectorT::value_type VectorMax(const VectorT& v)
{
    const typename VectorT::value maxVal = std::numeric_limits<typename VectorT::value_type>::min();

    for (std::size_t i = 0,
                     n = v.size();
         i < n;
         ++i)
    {
        if (maxVal < v[i])
        {
            maxVal = v[i];
        }
    }

    return maxVal;
}

template <typename VectorT>
typename VectorT::value_type VectorMin(const VectorT& v)
{
    const typename VectorT::value minVal = std::numeric_limits<typename VectorT::value_type>::max();

    for (std::size_t i = 0,
                     n = v.size();
         i < n;
         ++i)
    {
        if (minVal > v[i])
        {
            minVal = v[i];
        }
    }

    return minVal;
}

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMaxColumn(const MatrixT& A)
{
    const std::size_t nr = A.size();
    const std::size_t nc = A[0].size();

    std::vector<typename MatrixT::value_type> maxVals(nc, std::numeric_limits<typename MatrixT::value_type>::min());

    for (std::size_t j = 0; j < nc; ++j)
    {
        for (std::size_t i = 0; i < nr; ++i)
        {
            if (maxVals[j] < A[i][j])
            {
                maxVals[j] = A[i][j];
            }
        }
    }

    return maxVals;
}

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMinColumn(const MatrixT& A)
{
    const std::size_t nr = A.size();
    const std::size_t nc = A[0].size();

    std::vector<typename MatrixT::value_type> minVals(nc, std::numeric_limits<typename MatrixT::value_type>::max());

    for (std::size_t j = 0; j < nc; ++j)
    {
        for (std::size_t i = 0; i < nr; ++i)
        {
            if (minVals[j] > A[i][j])
            {
                minVals[j] = A[i][j];
            }
        }
    }

    return minVals;
}

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMaxRow(const MatrixT& A)
{
    const std::size_t nr = A.size();
    const std::size_t nc = A[0].size();

    std::vector<typename MatrixT::value_type> maxVals(nr, std::numeric_limits<typename MatrixT::value_type>::min());

    for (std::size_t i = 0; i < nr; ++i)
    {
        for (std::size_t j = 0; j < nc; ++j)
        {
            if (maxVals[i] < A[i][j])
            {
                maxVals[i] = A[i][j];
            }
        }
    }

    return maxVals;
}

template <typename MatrixT>
std::vector<typename MatrixT::value_type> MatrixMinRow(const MatrixT& A)
{
    const std::size_t nr = A.size();
    const std::size_t nc = A[0].size();

    std::vector<typename MatrixT::value_type> minVals(nr, std::numeric_limits<typename MatrixT::value_type>::max());

    for (std::size_t i = 0; i < nr; ++i)
    {
        for (std::size_t j = 0; j < nc; ++j)
        {
            if (minVals[i] > A[i][j])
            {
                minVals[i] = A[i][j];
            }
        }
    }

    return minVals;
}

template <typename MatrixT>
typename MatrixT::value_type MatrixMax(const MatrixT& A)
{
    return VectorMax(MatrixMaxRow(A));
}

template <typename MatrixT>
typename MatrixT::value_type MatrixMin(const MatrixT& A)
{
    return VectorMin(MatrixMinRow(A));
}

template <typename VectorT>
typename VectorT::value_type VectorNorm(VectorT& v, unsigned int p)
{
    if (p == 0)
    {
        FL_THROW2(std::invalid_argument, "In vector p-norm, the 'p' parameter must be a positive number");
    }

    typename VectorT::value_type norm = 0;

    if (p == 1)
    {
        for (std::size_t i = 0,
                         n = v.size();
             i < n;
             ++i)
        {
            norm += std::abs(v[i]);
        }
    }
    else if (p == 2)
    {
        for (std::size_t i = 0,
                         n = v.size();
             i < n;
             ++i)
        {
            norm += Sqr(std::abs(v[i]));
        }
        norm = std::sqrt(norm);
    }
    else
    {
        for (std::size_t i = 0,
                         n = v.size();
             i < n;
             ++i)
        {
            norm += std::pow(std::abs(v[i]), p);
        }
        norm = std::pow(norm, 1.0/p);
    }

    return norm;
}

template <typename InVectorT, typename OutVectorT>
OutVectorT& VectorCopy(const InVectorT& in, OutVectorT& out)
{
    out.resize(in.size());
    for (std::size_t i = 0,
                     n = in.size();
         i < n;
         ++i)
    {
        out[i] = in[i];
    }

    return out;
}

template <typename InMatrixT, typename OutMatrixT>
OutMatrixT& MatrixCopy(const InMatrixT& in, OutMatrixT& out)
{
    out.resize(in.size());
    for (std::size_t r = 0,
                     nr = in.size();
         r < nr;
         ++r)
    {
        out[r].resize(in[r].size());
        for (std::size_t c = 0,
                         nc = in[r].size();
             c < nc;
             ++c)
        {
            out[r][c] = in[r][c];
        }
    }

    return out;
}

template <typename T>
std::vector< std::vector<T> > MatrixZero(std::size_t nr, std::size_t nc)
{
    std::vector< std::vector<T> > A(nr);
    for (std::size_t i = 0; i < nr; ++i)
    {
        A[i].resize(nc, 0);
    }

    return A;
}

template <typename T>
std::vector<T> VectorZero(std::size_t n)
{
    return std::vector<T>(n, 0);
}


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
	int rank(RealT thresh = -1) const;

    /**
     * Return the nullity of A, after zeroing any singular values smaller than thresh.
     *
     * If positive, thresh is the threshold value below which singular values
     * are considered as zero.
     * If thresh is negative, a default based on expected roundoff error is used.
     */
	int nullity(RealT thresh = -1) const;

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
int SVDDecomposition<RealT>::rank(RealT thresh) const
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
int SVDDecomposition<RealT>::nullity(RealT thresh) const
{
    const RealT tsh = (thresh >= 0) ? thresh : this->getDefaultThreshold();

	int nn = 0;
	for (int j = 0; j < n_; ++j)
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

	int nr = 0;
	std::vector< std::vector<RealT> > rnge(m_, this->rank(thresh));
	for (int j = 0; j < n_; ++j)
    {
		if (w_[j] > tsh) {
			for (int i = 0; i < m_; ++i)
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

	int nn = 0;
    for (int j = 0; j < n_; ++j)
    {
        if (w_[j] <= tsh)
        {
            for (int jj = 0; jj < n_; ++jj)
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

}} // Namespace fl::detail


#endif // FL_DETAIL_MATH_H
/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */
