/**
 * \file fl/ann/activation_functions.h
 *
 * \brief Activation functions for neurons
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

#ifndef FL_ANN_ACTIVATION_FUNCTIONS_H
#define FL_ANN_ACTIVATION_FUNCTIONS_H


#include <boost/noncopyable.hpp>
#include <cmath>
#include <my/commons.h>


namespace fl { namespace ann {

/// Base class for activation functions
template <typename ValueT>
class ActivationFunction: boost::noncopyable
{
	public: virtual ~ActivationFunction() { }

	public: ValueT eval(ValueT x) const
	{
		return this->doEval(x);
	}

	public: ValueT evalDerivative(ValueT x) const
	{
		return this->doEvalDerivative(x);
	}

	public: std::pair<ValueT,ValueT> getOutputRange() const
	{
		return this->doGetOutputRange();
	}

	public: std::pair<ValueT,ValueT> getActiveInputRange() const
	{
		return this->doGetActiveInputRange();
	}

	protected: virtual std::pair<ValueT,ValueT> doGetOutputRange() const
	{
		return std::make_pair(this->eval(std::numeric_limits<ValueT>::min()),
							  this->eval(std::numeric_limits<ValueT>::max()));
	}

	protected: virtual std::pair<ValueT,ValueT> doGetActiveInputRange() const
	{
		return std::make_pair(this->eval(std::numeric_limits<ValueT>::min()),
							  this->eval(std::numeric_limits<ValueT>::max()));
	}

	private: virtual ValueT doEval(ValueT x) const = 0;

	private: virtual ValueT doEvalDerivative(ValueT x) const = 0;
}; // ActivationFunction

/**
 * Step activation function.
 *
 * The step activation function \f$f(x;t,h,l)\f$ is defined as:
 * \f[
 *  f(x;t,h,l) = \begin{cases}
 *                h, & x \ge t,
 *                l, & x < 0.
 *               \end{cases}
 * \f]
 * where \f$t\f$ is the threshold, \f$h\f$ is the value of the function when the
 * input \f$x\f$ is equal or above the threshold, and \f$l\f$ is the value of
 * the function when \f$l\f$ is below the threshold.
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class StepActivationFunction: public ActivationFunction<ValueT>
{
	public: explicit StepActivationFunction(ValueT low = 0,
											ValueT high = 1,
											ValueT threshold = 0)
	: l_(low),
	  h_(high),
	  t_(threshold)
	{
	}

	public: ValueT getThreshold() const
	{
		return t_;
	}

	public: ValueT getLowValue() const
	{
		return l_;
	}

	public: ValueT getHighValue() const
	{
		return h_;
	}

	private: ValueT doEval(ValueT x) const
	{
		if (x >= t_)
		{
			return h_;
		}
		return l_;
	}

	private: ValueT doEvalDerivative(ValueT x) const
	{
		if (x >= t_)
		{
			return h_;
		}
		return l_;
	}

	private: std::pair<ValueT,ValueT> doGetOutputRange() const
	{
		return std::make_pair(l_, h_);
	}

	private: std::pair<ValueT,ValueT> doGetActiveInputRange() const
	{
		return std::make_pair(t_, t_);
	}


	private: ValueT l_;
	private: ValueT h_;
	private: ValueT t_;
}; // StepActivationFunction

/**
 * Hard-limit activation function.
 *
 * The hard-limit activation function \f$f(x)\f$ is defined as:
 * \f[
 *  f(x) = \begin{cases}
 *           1, & x \ge 0,
 *           0, & x < 0.
 *         \end{cases}
 * \f]
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class HardLimitActivationFunction: public StepActivationFunction<ValueT>
{
	private: typedef StepActivationFunction<ValueT> BaseType;


	public: HardLimitActivationFunction()
	: BaseType(0, 1, 0)
	{
	}
}; // HardLimitActivationFunction

/**
 * Symmetric hard-limit activation function.
 *
 * The symmetric hard-limit activation function \f$f(x)\f$ is defined as:
 * \f[
 *  f(x) = \begin{cases}
 *           1, & x \ge 0,
 *           -1, & x < 0.
 *         \end{cases}
 * \f]
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class SymmetricHardLimitActivationFunction: public StepActivationFunction<ValueT>
{
	private: typedef StepActivationFunction<ValueT> BaseType;


	public: SymmetricHardLimitActivationFunction()
	: BaseType(1, -1, 0)
	{
	}
}; // SymmetricHardLimitActivationFunction

/**
 * Linear activation function.
 *
 * The linear activation function \f$f(x;k)\f$ is defined as:
 * \f[
 *  f(x;k) = kx
 * \f]
 * where \f$k\f$ is the slope.
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class LinearActivationFunction: public ActivationFunction<ValueT>
{
	public: explicit LinearActivationFunction(ValueT slope = 1)
	: k_(slope)
	{
	}

	public: ValueT getSlope() const
	{
		return k_;
	}

	private: ValueT doEval(ValueT x) const
	{
		return k_*x;
	}

	private: ValueT doEvalDerivative(ValueT x) const
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING(x);

		return k_;
	}


	private: ValueT k_;
}; // LinearActivationFunction

/**
 * Pure Linear activation function.
 *
 * The pure linear activation function \f$f(x;k)\f$ is defined as:
 * \f[
 *  f(x;k) = x
 * \f]
 *
 * This is a special case of the linear activation function with slope 1 and
 * intercept 0.
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class PureLinearActivationFunction: public LinearActivationFunction<ValueT>
{
	private: typedef LinearActivationFunction<ValueT> BaseType;


	public: PureLinearActivationFunction()
	: BaseType(1)
	{
	}
}; // PureLinearActivationFunction

/**
 * Logistic activation function.
 *
 * The logistic activation function \f$f(x;a,b,c)\f$ is defined as:
 * \f[
 *  f(x;l,s,o) = \frac{l}{1+e^{-s(x-o)}}
 * \f]
 * where \f$l\f$ is a positive constant called the upper limiting value, \f$s\f$ is a positive constant called the slope, and \f$o\f$ is a positive constant called the offset.
 *
 * The first derivative is:
 * \f[
 *  f'(x;l,s,o) = \frac{ls e^{-s(x-o)}}{(1+e^{-s(x-o)})^2} = \frac{s}{l}f(x;l,s,o)(l-f(x;l,s,o))
 * \f]
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class LogisticActivationFunction: public ActivationFunction<ValueT>
{
	public: explicit LogisticActivationFunction(ValueT limit = 1, ValueT slope = 1, ValueT offset = 0)
	: l_(limit),
	  s_(slope),
	  o_(offset)
	{
	}

	public: ValueT getUpperLimitingValue() const
	{
		return l_;
	}

	public: ValueT getSlope() const
	{
		return s_;
	}

	public: ValueT getOffset() const
	{
		return o_;
	}

	private: ValueT doEval(ValueT x) const
	{
		return l_/(1+std::exp(-s_*(x-o_)));
	}

	private: ValueT doEvalDerivative(ValueT x) const
	{
		const ValueT fx = this->eval(x);
		return fx*(l_-fx)*s_/l_;
	}

	private: std::pair<ValueT,ValueT> doGetOutputRange() const
	{
		return std::make_pair(0, l_);
	}

	private: std::pair<ValueT,ValueT> doGetActiveInputRange() const
	{
		return std::make_pair(-4/s_+o_, 4/s_+o_);
	}


	private: ValueT l_; ///< The limiting value parameter
	private: ValueT s_; ///< The slope parameter
	private: ValueT o_; ///< The offset parameter
}; // LogisticActivationFunction

/**
 * Logistic sigmoid activation function.
 *
 * The logistic sigmoid (Log-sigmoid) activation function \f$f(x)\f$ is defined
 * as:
 * \f[
 *  f(x) = \frac{1}{1+e^{-x}}
 * \f]
 * The Log-sigmoid function can be seen as a special case of the logistic
 * function \f$g(x;l,s,o)\f$ with parameters fixed to specific values, that is:
 * \f[
 *  f(x) := g(x;1,1,1)
 * \f]
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class LogSigmoidActivationFunction: public ActivationFunction<ValueT>
{
	//NOTE: we could inherit from LogisticActivationFunction but it would be much more inefficient in terms of computation

	private: ValueT doEval(ValueT x) const
	{
		return 1.0/(1+std::exp(-x));
	}

	private: ValueT doEvalDerivative(ValueT x) const
	{
		const ValueT fx = this->eval(x);
		return fx*(1-fx);
	}

	private: std::pair<ValueT,ValueT> doGetOutputRange() const
	{
		return std::make_pair(0, 1);
	}

	private: std::pair<ValueT,ValueT> doGetActiveInputRange() const
	{
		return std::make_pair(-4, 4);
	}
}; // LogSigmoidActivationFunction

/**
 * Bipolar sigmoid activation function.
 *
 * The bipolar sigmoid activation function \f$f(x)\f$
 * is defined as:
 * \f[
 *  f(x) = \frac{2}{1+e^{-x}}-1
 * \f]
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class BipolarSigmoidActivationFunction: public ActivationFunction<ValueT>
{
	private: ValueT doEval(ValueT x) const
	{
		return 2.0/(1+std::exp(-x))-1;
	}

	private: ValueT doEvalDerivative(ValueT x) const
	{
		const ValueT fx = this->eval(x);
		return 0.5*(1-fx*fx);
	}

	private: std::pair<ValueT,ValueT> doGetOutputRange() const
	{
		return std::make_pair(-1, 1);
	}

	private: std::pair<ValueT,ValueT> doGetActiveInputRange() const
	{
		return std::make_pair(0, 4);
	}
}; // BipolarSigmoidActivationFunction

/**
 * Hyperbolic tangent sigmoid activation function.
 *
 * The hyperbolic tangent sigmoid (Tan-sigmoid) activation function \f$f(x)\f$
 * is defined as:
 * \f[
 *  f(x) = \frac{2}{1+e^{-2x}}-1
 * \f]
 * The Tan-sigmoid is sometimes referred to as symmetric sigmoid function.
 * It is mathematically equivalent to the hyperbolic tangent:
 * \f[
 *  \mathrm{tanh}(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=\frac{1-e^{-2x}}{1+e^{-2x}}
 * \f]
 * 
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class TanSigmoidActivationFunction: public ActivationFunction<ValueT>
{
	private: ValueT doEval(ValueT x) const
	{
		//return 2.0/(1+std::exp(-2.0*x))-1;
		return std::tanh(x);
	}

	private: ValueT doEvalDerivative(ValueT x) const
	{
		const ValueT fx = this->eval(x);
		return 1-fx*fx;
	}

	private: std::pair<ValueT,ValueT> doGetOutputRange() const
	{
		return std::make_pair(-1, 1);
	}

	private: std::pair<ValueT,ValueT> doGetActiveInputRange() const
	{
		return std::make_pair(-2, 2);
	}
}; // TanSigmoidActivationFunction

}} // Namespace fl::ann

#endif // FL_ANN_ACTIVATION_FUNCTIONS_H
