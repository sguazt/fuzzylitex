#ifndef FL_DETAIL_RLS_H
#define FL_DETAIL_RLS_H


#include <cstddef>
#include <fl/detail/math.h>
#include <fl/commons.h>
#include <limits>
#include <stdexcept>
#include <vector>


namespace fl { namespace detail {

/**
 * Recursive Least Squares (RLS) estimator.
 *
 * Given a linear discrete-time system:
 * \f[
 *  y(n) = \sum_{k=0}^p \theta_k u(n-k)+v(n)
 * \f]
 * recursively estimates in time the parameters \f$\hat{\theta}_k(n)\f$ to
 * minimize the sum of squared errors
 * \f[
 *  \sum_{i=1}^n \lambda^{n-i}(y(n)-\hat{y}(n))^2
 * \f]
 * where \f$\hat{y}(n)=\sum_{k=0}^p \theta_k u(n-k)\f$.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class RecursiveLeastSquaresEstimator
{
	private: typedef std::vector<ValueT> VectorType;
	private: typedef std::vector< std::vector<ValueT> > MatrixType;


	/// Constructs a RLS estimator where \a na is the filter order, \a ni is the input dimension, and \a no is the output dimension
	public: RecursiveLeastSquaresEstimator(std::size_t p, std::size_t nu, std::size_t ny, ValueT lambda)
	: p_(p),
	  nu_(nu),
	  ny_(ny),
	  lambda_(lambda),
	  count_(0)
	{
		this->reset();
	}

	public: void reset(ValueT delta = 1e+4)
	{
		//const std::size_t n = ny_+p_*nu_;
		const std::size_t n = p_*nu_;

		phi_.clear();
		phi_.resize(n, 0);

		Theta_.clear();
		Theta_.resize(n);
		P_.clear();
		P_.resize(n);
		for (std::size_t i = 0; i < n; ++i)
		{
			Theta_[i].resize(ny_, 0);
			//Theta_[i].resize(ny_, std::numeric_limits<ValueT>::epsilon());
			P_[i].resize(n, 0);
			P_[i][i] = delta;
		}

		count_ = 0;
	}

	public: template <typename UIterT, typename YIterT>
			std::vector<ValueT> estimate(UIterT uFirst, UIterT uLast, YIterT yFirst, YIterT yLast)
	{
		return this->estimate(VectorType(uFirst, uLast), VectorType(yFirst, yLast));
	}

	public: std::vector<ValueT> estimate(const std::vector<ValueT>& u, const std::vector<ValueT>& y)
	{
		if (u.size() != nu_)
		{
			FL_THROW2(std::invalid_argument, "Input dimension does not match");
		}
		if (y.size() != ny_)
		{
			FL_THROW2(std::invalid_argument, "Output dimension does not match");
		}

		++count_;

		const std::size_t n = phi_.size();

std::cerr << "phi(k)="; fl::detail::VectorOutput(std::cerr, phi_); std::cerr << std::endl; //XXX
std::cerr << "P(k)="; fl::detail::MatrixOutput(std::cerr, P_); std::cerr << std::endl; //XXX
std::cerr << "Theta(k)="; fl::detail::MatrixOutput(std::cerr, Theta_); std::cerr << std::endl; //XXX
	VectorType yhat;

if (count_ > p_)
{
		// Compute the Gain:
		//  $l(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}$
		const VectorType l = fl::detail::VectorScalarProduct(
								fl::detail::MatrixVectorProduct(P_, phi_),
								1.0/(lambda_
									 + fl::detail::VectorInnerProduct(
									   fl::detail::VectorMatrixProduct(phi_, P_), phi_)));
std::cerr << "l="; fl::detail::VectorOutput(std::cerr, l); std::cerr << std::endl; //XXX

		// Update the covariance matrix by means of the matrix inversion lemma (use the Woodbury's identity: A=B^{-1}+CD^{-1}C^T ==> A^{-1}=B-BC(D+C^TBC)^{-1}C^TB)
		//  $P(k+1) = \frac{1}{\lambda(k)}\left[I-l(k+1)\Phi^T(k+1)\right]P(k)$
		P_ = fl::detail::MatrixScalarProduct(
				fl::detail::MatrixProduct(
					fl::detail::MatrixDiff(
						fl::detail::MatrixIdentity<MatrixType>(n, n),
						fl::detail::VectorOuterProduct<MatrixType>(l, phi_)),
					P_),
				1.0/lambda_);

		// Computes the output estimate
		//  $\hat{y}(k) = (\phi^T(k)\Theta(k))^T = \Theta^T(k)\phi(k)$
		yhat = fl::detail::VectorMatrixProduct(phi_, Theta_);

		// Update parameters estimate
		//  $\hat{\Theta}(k+1) = \hat{\Theta}(k)+l^T(k+1)(y^T(k+1)-\phi^T(k+1)\hat{\Theta}(k))$
		Theta_ = fl::detail::MatrixSum(
					Theta_,
					fl::detail::VectorOuterProduct<MatrixType>(
						l,
						fl::detail::VectorDiff(y, yhat)));
}
else
{
	yhat.resize(ny_, 0);
}

		// Update the regressor vector
//		//  $\phi(k+1) = [y_1(k) ... y_1(k-n_a+1) ... y_{n_y}(k) ... y_{n_y}(k-n_a+1)]^T$
//		VectorType phiNew(n, 0);
//		// - Copy the new y into the regressor vector: \phi(k+1) = [y_1(k) 0 ... 0 y_2(k) 0 ... 0 y_{n_y}(k) 0 ... 0]^T
//		for (std::size_t i = 0; i < ny_; ++i)
//		{
//			phiNew[i*p_] = y[i];
//		}
//		// - Append the old na-1 y values into the regressor vector: \phi(k+1) = [y_1(k) y_1(k-1) ... y_1(k-n_a+1) y_2(k) y_2(k-1) ... y_{n_y}(k-n_a+1)]^T
//		for (std::size_t i = 1; i < p_; ++i)
//		{
//			for (std::size_t j = 0; j < nu_; ++j)
//			{
//				const std::size_t jna = j*p_;
//
//				phiNew[i+jna] = phi_[(i-1)+jna];
//			}
//		}
		//  $\phi(k+1) = [u_1(k) ... u_1(k-p+1) ... u_{n_u}(k) ... u_{n_u}(k-p+1)]^T$
		VectorType phiNew(n, 0);
		// - Copy the new u into the regressor vector: \phi(k+1) = [u_1(k) 0 ... 0 u_2(k) 0 ... 0 u_{n_u}(k) 0 ... 0]^T
		for (std::size_t i = 0; i < nu_; ++i)
		{
			phiNew[i*p_] = u[i];
		}
		// - Append the old na-1 y values into the regressor vector: \phi(k+1) = [u_1(k) u_1(k-1) ... u_1(k-p+1) u_2(k) u_2(k-1) ... u_{n_u}(k-p+1)]^T
		for (std::size_t i = 1; i < p_; ++i)
		{
			for (std::size_t j = 0; j < nu_; ++j)
			{
				const std::size_t jna = j*p_;

				phiNew[i+jna] = phi_[(i-1)+jna];
			}
		}

		phi_ = phiNew;

std::cerr << "phi(k+1)="; fl::detail::VectorOutput(std::cerr, phi_); std::cerr << std::endl; //XXX
std::cerr << "P(k+1)="; fl::detail::MatrixOutput(std::cerr, P_); std::cerr << std::endl; //XXX
std::cerr << "Theta(k+1)="; fl::detail::MatrixOutput(std::cerr, Theta_); std::cerr << std::endl; //XXX

		return yhat;
	}


	private: std::size_t p_; ///< The model order
	private: std::size_t nu_; ///< The input dimension
	private: std::size_t ny_; ///< The output dimension
	private: ValueT lambda_; ///< Forgetting factor
	private: MatrixType Theta_; ///< Parameter matrix
	private: MatrixType P_; ///< Covariance matrix
	private: VectorType phi_; ///< Regressor vector
	private: std::size_t count_;
}; // RecursiveLeastSquares

}} // Namespace fl::detail

#endif // FL_DETAIL_RLS_H
