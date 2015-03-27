#ifndef FL_DETAIL_MATH_HPP
#define FL_DETAIL_MATH_HPP


#include <cstddef>
#include <fl/commons.h>
#include <iostream>
#include <vector>


namespace fl { namespace detail {

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

/// Returns evenly \a spaced values over the given interval [\a start, \a stop]. The end-point of the interval can be optionally excluded.
template <typename ValueT>
std::vector<ValueT> LinSpace(ValueT start, ValueT stop, std::size_t num, bool endPoint = true)
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
}

template <typename CharT, typename CharTraitsT, typename MatrixT>
void MatrixOutput(std::basic_ostream<CharT,CharTraitsT>& os, const MatrixT& A)
{
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
}

/// Computes v*c, where v is a nx1 vector and c is a scalar value
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

/// Computes A*c, where A is a nxm matrix and c is a scalar value
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

/// Computes u+v, where u and v are nx1 vectors
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

/// Computes u+v, where u and v are nx1 vectors
template <typename VectorT>
VectorT VectorSum(const VectorT& u, const VectorT& v)
{
	return VectorAlgebraicSum(u, v, false);
}

/// Computes u-v, where u and v are nx1 vectors
template <typename VectorT>
VectorT VectorDiff(const VectorT& u, const VectorT& v)
{
	return VectorAlgebraicSum(u, v, true);
}

/// Computes the transpose A' of A, where A is a nxm matrix
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

/// Computes the transpose A' of A, where A is a nxm matrix
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

/// Computes A+B, where A and B are nxm matrices
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

/// Computes A+B, where A and B are nxm matrices
template <typename MatrixT>
MatrixT MatrixSum(const MatrixT& A, const MatrixT& B)
{
	return MatrixAlgebraicSum(A, B, false);
}

/// Computes A-B, where A and B are nxm matrices
template <typename MatrixT>
MatrixT MatrixDiff(const MatrixT& A, const MatrixT& B)
{
	return MatrixAlgebraicSum(A, B, true);
}

/// Computes A*B, where A is a nxp matrix and B is a pxm matrix
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

/// Computes the inner product u'*v, where u and v are nx1 vectors, and u' is the transpose of u
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
 
/// Computes the outer product u*v', where u and v are nx1 vectors, and v' is the transpose of v
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
 
/// Computes A*v, where A is a nxp matrix and v is a px1 vector
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

/// Computes v'*A, wherev is a px1 vector,  A is a pxn matrix, and v' is the transpose of v
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

}} // Namespace fl::detail


#endif // FL_DETAIL_MATH_HPP
