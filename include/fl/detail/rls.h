/**
 * \file fl/detail/rls.h
 *
 * \brief The recursive least-squares algorithm
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

#ifndef FL_DETAIL_RLS_H
#define FL_DETAIL_RLS_H


#include <cstddef>
#include <fl/detail/math.h>
#include <fl/fuzzylite.h>
#include <fl/macro.h>
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
 * \tparam ValueT The type for floating-point numbers
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class FL_API RecursiveLeastSquaresEstimator
{
private:
    typedef std::vector<ValueT> VectorType; ///< The vector type
    typedef std::vector< std::vector<ValueT> > MatrixType; ///< The matrix type


public:
    /// Default constructor
    RecursiveLeastSquaresEstimator();

    /// Constructs a RLS estimator where \a na is the filter order, \a ni is the input dimension, and \a no is the output dimension
    RecursiveLeastSquaresEstimator(std::size_t p, std::size_t nu, std::size_t ny, ValueT lambda);

    /// Sets the order of the model
    void setModelOrder(std::size_t order);

    /// Gets the order of the model
    std::size_t getModelOrder() const;

    /// Sets the size of the input variable
    void setInputDimension(std::size_t n);

    /// Gets the size of the input variable
    std::size_t getInputDimension() const;

    /// Sets the size of the output variable
    void setOutputDimension(std::size_t n);

    /// Gets the size of the output variable
    std::size_t getOutputDimension() const;

    /// Sets the forgetting factor
    void setForgettingFactor(ValueT lambda);

    /// Gets the forgetting factor
    ValueT getForgettingFactor() const;

    // Sets the inverse of the pseudo covariance matrix
    //void setCovarianceInverse(const std::vector< std::vector<ValueT> >& P);

    /// Gets the inverse of the pseudo covariance matrix
    std::vector< std::vector<ValueT> > getCovarianceInverse() const;

    // Sets the regressor vector
    //void setRegressor(const std::vector<ValueT>& phi);

    /// Gets the regressor vector
    std::vector<ValueT> getRegressor() const;

    /// Gets the matrix of estimated parameters
    std::vector< std::vector<ValueT> > getEstimatedParameters() const;

    /// Gets the total number of iteration performed to date
    std::size_t numberOfIterations() const;

    /// Resets/initialize the algorithm
    void reset(ValueT delta = 1.0e+6);

    /// Performs an iteration of the RLS algorithm with respect to the given inputs and outputs, and returns the estimated output
    template <typename UIterT, typename YIterT>
    std::vector<ValueT> estimate(UIterT uFirst, UIterT uLast, YIterT yFirst, YIterT yLast);

    /// Performs an iteration of the RLS algorithm with respect to the given inputs and outputs, and returns the estimated output
    std::vector<ValueT> estimate(const std::vector<ValueT>& u, const std::vector<ValueT>& y);


private:
    std::size_t p_; ///< The model order
    std::size_t nu_; ///< The input dimension
    std::size_t ny_; ///< The output dimension
    ValueT lambda_; ///< Forgetting factor
    MatrixType Theta_; ///< Parameter matrix
    MatrixType P_; ///< Covariance matrix
    VectorType phi_; ///< Regressor vector
    std::size_t count_; ///< The total number of iterations performed so far
}; // RecursiveLeastSquares


////////////////////////
// Template definitions
////////////////////////


template <typename ValueT>
RecursiveLeastSquaresEstimator<ValueT>::RecursiveLeastSquaresEstimator()
: p_(0),
  nu_(0),
  ny_(0),
  lambda_(0),
  count_(0)
{
}

template <typename ValueT>
RecursiveLeastSquaresEstimator<ValueT>::RecursiveLeastSquaresEstimator(std::size_t p, std::size_t nu, std::size_t ny, ValueT lambda)
: p_(p+1),
  nu_(nu),
  ny_(ny),
  lambda_(lambda),
  count_(0)
{
    this->reset();
}

template <typename ValueT>
void RecursiveLeastSquaresEstimator<ValueT>::setModelOrder(std::size_t order)
{
    p_ = order+1;
}

template <typename ValueT>
std::size_t RecursiveLeastSquaresEstimator<ValueT>::getModelOrder() const
{
	if (p_ == 0)
	{
		return 0;
	}

    return p_-1;
}

template <typename ValueT>
void RecursiveLeastSquaresEstimator<ValueT>::setInputDimension(std::size_t n)
{
    nu_ = n;
}

template <typename ValueT>
std::size_t RecursiveLeastSquaresEstimator<ValueT>::getInputDimension() const
{
    return nu_;
}

template <typename ValueT>
void RecursiveLeastSquaresEstimator<ValueT>::setOutputDimension(std::size_t n)
{
    ny_ = n;
}

template <typename ValueT>
std::size_t RecursiveLeastSquaresEstimator<ValueT>::getOutputDimension() const
{
    return ny_;
}

template <typename ValueT>
void RecursiveLeastSquaresEstimator<ValueT>::setForgettingFactor(ValueT lambda)
{
    lambda_ = lambda;
}

template <typename ValueT>
ValueT RecursiveLeastSquaresEstimator<ValueT>::getForgettingFactor() const
{
    return lambda_;
}

//template <typename ValueT>
//void RecursiveLeastSquaresEstimator<ValueT>::setCovarianceInverse(const std::vector< std::vector<ValueT> >& P)
//{
//  P_ = P;
//}

template <typename ValueT>
std::vector< std::vector<ValueT> > RecursiveLeastSquaresEstimator<ValueT>::getCovarianceInverse() const
{
    return P_;
}

//template <typename ValueT>
//void RecursiveLeastSquaresEstimator<ValueT>::setRegressor(const std::vector<ValueT>& phi)
//{
//  phi_ = phi;
//}

template <typename ValueT>
std::vector<ValueT> RecursiveLeastSquaresEstimator<ValueT>::getRegressor() const
{
    return phi_;
}

template <typename ValueT>
std::vector< std::vector<ValueT> > RecursiveLeastSquaresEstimator<ValueT>::getEstimatedParameters() const
{
    return Theta_;
}

template <typename ValueT>
std::size_t RecursiveLeastSquaresEstimator<ValueT>::numberOfIterations() const
{
    return count_;
}

template <typename ValueT>
void RecursiveLeastSquaresEstimator<ValueT>::reset(ValueT delta)
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

template <typename ValueT>
template <typename UIterT, typename YIterT>
std::vector<ValueT> RecursiveLeastSquaresEstimator<ValueT>::estimate(UIterT uFirst, UIterT uLast, YIterT yFirst, YIterT yLast)
{
    return this->estimate(VectorType(uFirst, uLast), VectorType(yFirst, yLast));
}

template <typename ValueT>
std::vector<ValueT> RecursiveLeastSquaresEstimator<ValueT>::estimate(const std::vector<ValueT>& u, const std::vector<ValueT>& y)
{
    if (u.size() != nu_)
    {
        FL_THROW2(std::invalid_argument, "Input dimension does not match");
    }
    if (y.size() != ny_)
    {
        FL_THROW2(std::invalid_argument, "Output dimension does not match");
    }
    if (nu_ == 0)
    {
        FL_THROW2(std::logic_error, "Wrong input dimension");
    }
    if (ny_ == 0)
    {
        FL_THROW2(std::logic_error, "Wrong output dimension");
    }
    if (p_ == 0)
    {
        FL_THROW2(std::logic_error, "Wrong model order");
    }

    ++count_;

    const std::size_t n = phi_.size();

//std::cerr << "COUNT=" << count_ << std::endl;//XXX
//std::cerr << "y(k)="; fl::detail::VectorOutput(std::cerr, y); std::cerr << std::endl; //XXX
//std::cerr << "phi(k)="; fl::detail::VectorOutput(std::cerr, phi_); std::cerr << std::endl; //XXX
//std::cerr << "P(k)="; fl::detail::MatrixOutput(std::cerr, P_); std::cerr << std::endl; //XXX
//std::cerr << "Theta(k)="; fl::detail::MatrixOutput(std::cerr, Theta_); std::cerr << std::endl; //XXX
    VectorType yhat;

    // Update the regressor vector
    //  $\phi(k+1) = [u_1(k) ... u_1(k-p+1) ... u_{n_u}(k) ... u_{n_u}(k-p+1)]^T$
    VectorType phiNew(n, 0);
    // - Copy the new u into the regressor vector: \phi(k+1) = [u_1(k) 0 ... 0 u_2(k) 0 ... 0 u_{n_u}(k) 0 ... 0]^T
    for (std::size_t i = 0; i < nu_; ++i)
    {
        phiNew[i*p_] = u[i];
    }
    // - Append the old na-1 u values into the regressor vector: \phi(k+1) = [u_1(k) u_1(k-1) ... u_1(k-p+1) u_2(k) u_2(k-1) ... u_{n_u}(k-p+1)]^T
    for (std::size_t i = 1; i < p_; ++i)
    {
        for (std::size_t j = 0; j < nu_; ++j)
        {
            const std::size_t jna = j*p_;

            phiNew[i+jna] = phi_[(i-1)+jna];
        }
    }
    phi_ = phiNew;

    // Update parameter and covariance matrices (to be done only after enough observations have been seen)
    if (count_ >= p_)
    {
#if 0
            // Compute the Gain:
            //  $l(k+1) = \frac{P(k)\phi(k+1)}{\lambda(k)+\phi^T(k+1)P(k)\phi(k+1)}$
            const VectorType l = fl::detail::VectorScalarProduct(
                                    fl::detail::MatrixVectorProduct(P_, phi_),
                                    1.0/(lambda_ + fl::detail::VectorInnerProduct(fl::detail::VectorMatrixProduct(phi_, P_), phi_)));

            // Update the covariance matrix by means of the matrix inversion lemma (use the Woodbury's identity: A=B^{-1}+CD^{-1}C^T ==> A^{-1}=B-BC(D+C^TBC)^{-1}C^TB)
            //  $P(k+1) = \frac{1}{\lambda(k)}\left[I-l(k+1)\Phi^T(k+1)\right]P(k)$
            //P_ = fl::detail::MatrixScalarProduct(fl::detail::MatrixDiff(P_, fl::detail::MatrixScalarProduct(fl::detail::MatrixProduct(fl::detail::VectorOuterProduct<MatrixType>(fl::detail::MatrixVectorProduct(P_, phi_), phi_), P_), 1.0/(lambda_+fl::detail::VectorInnerProduct(fl::detail::VectorMatrixProduct(phi_, P_), phi_)))), 1.0/lambda_);
            P_ = fl::detail::MatrixScalarProduct(
                    fl::detail::MatrixProduct(
                        fl::detail::MatrixDiff(
                            fl::detail::MatrixIdentity<MatrixType>(n, n),
                            fl::detail::VectorOuterProduct<MatrixType>(l, phi_)),
                        P_),
                    1.0/lambda_);

            // Computes the output estimate
            //  $\hat{y}(k) = (\phi^T(k)\Theta(k))^T$
            yhat = fl::detail::VectorMatrixProduct(phi_, Theta_);

            // Update parameters estimate
            //  $\hat{\Theta}(k+1) = \hat{\Theta}(k)+l^T(k+1)(y^T(k+1)-\phi^T(k+1)\hat{\Theta}(k))$
            Theta_ = fl::detail::MatrixSum(
                        Theta_,
                        fl::detail::VectorOuterProduct<MatrixType>(
                            l,
                            fl::detail::VectorDiff(y, yhat)));
#else // if 0
            // Update the covariance matrix by means of the matrix inversion lemma (use the Woodbury's identity: A=B^{-1}+CD^{-1}C^T ==> A^{-1}=B-BC(D+C^TBC)^{-1}C^TB)
            //  $P(k+1) = \frac{1}{\lambda(k)}\left[P(k)-\frac{P(k)\phi(k+1)\phi^T(k+1)P(k)}{\lambda+\phi^T(k+1)P(k)\phi(k+1)}\right]$
/*
            P_ = fl::detail::MatrixScalarProduct(
                        fl::detail::MatrixDiff(
							P_,
							fl::detail::MatrixScalarProduct(
								fl::detail::MatrixProduct(
									fl::detail::VectorOuterProduct<MatrixType>(
										fl::detail::MatrixVectorProduct(P_, phi_),
										phi_
									),
									P_
								),
								1.0/(lambda_+fl::detail::VectorInnerProduct(fl::detail::VectorMatrixProduct(phi_, P_), phi_))
							)
						),
						1.0/lambda_);
*/
			VectorType aux1 = fl::detail::MatrixVectorProduct(P_, phi_);
			VectorType aux2 = fl::detail::VectorMatrixProduct(phi_, P_);
			MatrixType aux3 = fl::detail::VectorOuterProduct<MatrixType>(aux1, aux2);
			const ValueT denom = lambda_ + fl::detail::VectorInnerProduct(phi_, aux1);
			aux3 = fl::detail::MatrixScalarProduct(aux3, 1.0/denom);
			P_ = fl::detail::MatrixDiff(P_, aux3);
			P_ = fl::detail::MatrixScalarProduct(P_, 1.0/lambda_);

            // Computes the output estimate
            //  $\hat{y}(k+1) = (\phi^T(k+1)\Theta(k))^T$
            yhat = fl::detail::VectorMatrixProduct(phi_, Theta_);

            // Update parameters estimate
            //  $\hat{\Theta}(k+1) = \hat{\Theta}(k)+P(k+1)\phi(k+1)[y^T(k+1)-\phi^T(k+1)\hat{\Theta}(k)]$
			////VectorType aux5 = fl::detail::VectorMatrixProduct(phi_, Theta_);
			////aux5 = fl::detail::VectorDiff(y, tmp5);
			//VectorType aux5 = fl::detail::VectorDiff(y, yhat);
			//const MatrixType aux6 = fl::detail::VectorOuterProduct<MatrixType>(phi_, aux5);
			//const MatrixType aux7 = fl::detail::MatrixProduct(P_, aux6);
			//Theta_ = fl::detail::MatrixSum(Theta_, aux7);
			aux1 = fl::detail::VectorDiff(y, yhat);
			aux3 = fl::detail::VectorOuterProduct<MatrixType>(phi_, aux1);
			MatrixType aux4 = fl::detail::MatrixProduct(P_, aux3);
			Theta_ = fl::detail::MatrixSum(Theta_, aux4);
#endif // if 0
//std::cerr << "RLS - Covariance Matrix ="; fl::detail::MatrixOutput(std::cerr, P_); std::cerr << std::endl; //XXX
//std::cerr << "RLS - Regressor = "; fl::detail::VectorOutput(std::cerr, phi_); std::cerr << std::endl; //XXX
//std::cerr << "RLS - Parameters Matrix ="; fl::detail::MatrixOutput(std::cerr, Theta_); std::cerr << std::endl; //XXX
//std::cerr << "RLS - yhat(k)="; fl::detail::VectorOutput(std::cerr, yhat); std::cerr << std::endl; //XXX
    }
    else
    {
        yhat.resize(ny_, std::numeric_limits<ValueT>::quiet_NaN());
    }

//std::cerr << "phi(k+1)="; fl::detail::VectorOutput(std::cerr, phi_); std::cerr << std::endl; //XXX
//std::cerr << "P(k+1)="; fl::detail::MatrixOutput(std::cerr, P_); std::cerr << std::endl; //XXX
//std::cerr << "Theta(k+1)="; fl::detail::MatrixOutput(std::cerr, Theta_); std::cerr << std::endl; //XXX

    return yhat;
}

}} // Namespace fl::detail

#endif // FL_DETAIL_RLS_H
