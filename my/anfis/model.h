#ifndef FL_ANFIS_MODEL_H
#define FL_ANFIS_MODEL_H

#include <fl/ann.h>
#include <fl/commons.h>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/Headers.h>


namespace fl { anfis {

namespace detail {

/**
 * Eval the derivative of the generalized bell function with respect to its parameters
 */
fl::scalar EvalBellTermDerivativeWrtParameter(fl::scalar x, unsigned short param)
{
	if (param > 2)
	{
		FL_THROW2(std::invalid_argument, "Unknown parameter for the generalizd bell-shaped membership function");
	}

	const fl::scalar w = p_specTerm->getWeight();
	const fl::scalar c = p_specTerm->getCenter();
	const fl::scalar s = p_specTerm->getSlope();

	const fl::scalar xn = (x-c)/w;
	const fl::scalar xnp = xn != 0 ? std::pow(xn*xn, s) : 0;
	const fl::scalar den = detail::sqr(1+xnp);

	switch (param)
	{
		case 0: // Weight parameter
			return 2.0*s*xnp/(w*den);
		case 1: // Slope parameter
			return (x != c && x != (c+w))
				   ? -std::log(detail::sqr(xn)*xnp/den
				   : 0;
		case 2: // Center parameter
			return (x != c)
				   ? 2.0*s*xnp/((x-c)*den)
				   : 0;
	}

	// Should never be executed. Just to quiet the compiler
	return fl::nan;
}

} // Namespace detail


////////////////////////////////////////////////////////////////////////////////
// Net Input Functions


/// Net input function based on fuzzy T-norm/S-norm operators
class NormNetInputFunction: public fl::ann::NetInputFunction<fl::scalar>
{
	private: typedef fl::ann::NetInputFunction<fl::scalar> BaseType;
	protected: typedef typename BaseType::ForwardIterator ForwardIterator;

	public: NormNetInputFunction(fl::Norm* p_norm)
	: p_norm_(p_norm)
	{
	}

	private: fl::scalar doEval(ForwardIterator inputFirst, ForwardIterator inputLast,
							   ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		fl::scalar res = fl::NaN;

		if (inputFirst != inputLast && weightFirst != weightLast)
		{
			res = *inputFirst++ * *weightFirst++;

			while (inputFirst != inputLast && weightFirst != weightLast)
			{
				const fl::scalar in = *inputFirst;
				const fl::scalar w = *weightFirst;
				const fl::scalar rhs = w*in;

				res = p_norm_->compute(res, rhs);

				++inputFirst;
				++weightFirst;
			}
		}

		return res;
	}

	private: std::vector<fl::scalar> doEvalDerivativeWrtInput(ForwardIterator inputFirst, ForwardIterator inputLast,
															  ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		if (dynamic_cast<fl::Minimum*>(p_norm_)
			|| dynamic_cast<fl::Maximum*>(p_norm_))
		{
			// The derivative is 0 for all inputs, excepts for the one
			// corresponding to the min (or max) for which the derivative is the
			// associated weight

			const fl::scalar fx = this->eval(inputFirst, inputLast, weightFirst, weightLast);

			std::vector<fl::scalar> res;
			while (inputFirst != inputLast && weightFirst != weightLast)
			{
				const fl::scalar in = *inputFirst;
				const fl::scalar w = *weightFirst;
				const fl::scalar val = w*in;

				if (fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(fx, val))
				{
					res.push_back(w);
				}
				else
				{
					res.push_back(0);
				}

				++inputFirst;
				++weightFirst;
			}

			return res;
		}
		else if (dynamic_cast<fl::AlgebraicProduct*>(p_norm_))
		{
			// The derivative for a certain input i is given by the product of
			// all the weights and inputs but i

			std::vector<fl::scalar> res;

			ForwardIterator inputIt = inputFirst;
			ForwardIterator weightIt = weightFirst;
			while (inputFirst != inputLast && weightFirst != weightLast)
			{
				fl::scalar prod = 1;
				while (inputIt != inputLast && weightIt != weightLast)
				{
					const fl::scalar w = *weightIt;

					if (inputIt != inputFirst)
					{
						const fl::scalar in = *inputIt;

						prod *= in
					}
					prod *= w;

					++inputIt;
					++weightIt;
				}

				res.push_back(prod);

				++inputFirst;
				++weightFirst;
			}

			return res;
		}
		else if (dynamic_cast<fl::AlgebraicSum*>(p_norm_))
		{
			// The derivative of each input is given by:
			//  \frac{\partial [x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3 + x_4 - (x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3)*x_4 + ...]}{\partial x_1}
			//  = 1 - x_2 - x_3 - x_2*x_3 - x_4 - x_2*x_4 - x_3*x_4 - x_2*x_3*x_4 + ...
			//  = (1-x2)*(1-x3)*(1-x4)*...
			// In order to keep into account weights we have:
			//  \frac{\partial [w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2) + w_3*x_3 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2))*w_3*x_3 + w_4*x_4 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2) + w_3*x_3 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2))*w_3*x_3)*w_4*x_4 + ...]}{\partial x_1}
			//  = w_1*(1-w_2*x_2)*(1-w_3*x_3)*(1-w_4*x_4)*...

			std::vector<fl::scalar> res;

			ForwardIterator inputIt = inputFirst;
			ForwardIterator weightIt = weightFirst;
			while (inputFirst != inputLast && weightFirst != weightLast)
			{
				fl::scalar prod = 1;
				while (inputIt != inputLast && weightIt != weightLast)
				{
					const fl::scalar w = *weightIt;

					if (inputIt != inputFirst)
					{
						const fl::scalar in = *inputIt;

						prod *= (1-in);
					}
					prod *= w;

					++inputIt;
					++weightIt;
				}

				res.push_back(prod);

				++inputFirst;
				++weightFirst;
			}

			return res;
		}
		else
		{
			FL_THROW2(std::runtime_error, "Norm operator '" + p_norm_->className() + "' not yet implemented");
		}
	}

	private: std::vector<fl::scalar> doEvalDerivativeWrtWeight(ForwardIterator inputFirst, ForwardIterator inputLast,
															   ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		if (dynamic_cast<fl::Minimum*>(p_norm_)
			|| dynamic_cast<fl::Maximum*>(p_norm_))
		{
			// The derivative is 0 for all inputs, excepts for the one
			// corresponding to the min (or max) for which the derivative is the
			// associated input value

			const fl::scalar fx = this->eval(inputFirst, inputLast, weightFirst, weightLast);

			std::vector<fl::scalar> res;
			while (inputFirst != inputLast && weightFirst != weightLast)
			{
				const fl::scalar in = *inputFirst;
				const fl::scalar w = *weightFirst;
				const fl::scalar val = w*in;

				if (fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(fx, val))
				{
					res.push_back(in);
				}
				else
				{
					res.push_back(0);
				}

				++inputFirst;
				++weightFirst;
			}

			return res;
		}
		else if (dynamic_cast<fl::AlgebraicProduct*>(p_norm_))
		{
			// The derivative for a certain input i is given by the product of
			// all the weights and inputs but i

			std::vector<fl::scalar> res;

			ForwardIterator inputIt = inputFirst;
			ForwardIterator weightIt = weightFirst;
			while (inputFirst != inputLast && weightFirst != weightLast)
			{
				fl::scalar prod = 1;
				while (inputIt != inputLast && weightIt != weightLast)
				{
					const fl::scalar in = *inputIt;

					if (weightIt != weightFirst)
					{
						const fl::scalar w = *weightIt;

						prod *= w
					}
					prod *= in;

					++inputIt;
					++weightIt;
				}

				res.push_back(prod);

				++inputFirst;
				++weightFirst;
			}

			return res;
		}
		else if (dynamic_cast<fl::AlgebraicSum*>(p_norm_))
		{
			// The derivative of each input is given by:
			//  \frac{\partial [x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3 + x_4 - (x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3)*x_4 + ...]}{\partial x_1}
			//  = 1 - x_2 - x_3 - x_2*x_3 - x_4 - x_2*x_4 - x_3*x_4 - x_2*x_3*x_4 + ...
			//  = (1-x2)*(1-x3)*(1-x4)*...
			// In order to keep into account weights we have:
			//  \frac{\partial [w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2) + w_3*x_3 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2))*w_3*x_3 + w_4*x_4 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2) + w_3*x_3 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2))*w_3*x_3)*w_4*x_4 + ...]}{\partial x_1}
			//  = w_1*(1-w_2*x_2)*(1-w_3*x_3)*(1-w_4*x_4)*...

			std::vector<fl::scalar> res;

			ForwardIterator inputIt = inputFirst;
			ForwardIterator weightIt = weightFirst;
			while (inputFirst != inputLast && weightFirst != weightLast)
			{
				fl::scalar prod = 1;
				while (inputIt != inputLast && weightIt != weightLast)
				{
					const fl::scalar in = *inputFirst;

					if (weightIt != weightFirst)
					{
						const fl::scalar w = *weightFirst;

						prod *= (1-w);
					}
					prod *= in;

					++inputIt;
					++weightIt;
				}

				res.push_back(prod);

				++inputFirst;
				++weightFirst;
			}

			return res;
		}
		else
		{
			FL_THROW2(std::runtime_error, "Norm operator '" + p_norm_->className() + "' not yet implemented");
		}
	}


	private: fl::Norm* p_norm_;
}; // NormNetInputFunction

////////////////////////////////////////////////////////////////////////////////
// Activation Functions


/// Activation function based on fuzzy terms
class TermActivationFunction: public fl::ann::ActivationFunction<fl::scalar>
{
	public: TermActivationtFunction(fl::Term* p_term)
	: p_term_(p_term)
	{
	}

	public: fl::Term* getTerm() const
	{
		return p_term_;
	}

	private: fl::scalar doEval(fl::scalar x) const
	{
		return p_term_->membership(x);
	}

	private: fl::scalar doEvalDerivative(fl::scalar x) const
	{
		FL_THROW2(std::runtime_error, "Derivative of a term MF should never be called");
	}


	private: fl::Term* p_term_;
}; // TermActivationFunction

/// Activation function based on fuzzy hedges
class HedgeActivationFunction: public fl::ann::ActivationFunction<fl::scalar>
{
	public: HedgeActivationtFunction(fl::Hedge* p_hedge)
	: p_hedge_(p_hedge)
	{
	}

	public: fl::Hedge* getHedge() const
	{
		return p_hedge_;
	}

	private: fl::scalar doEval(fl::scalar x) const
	{
		return p_hedge_->hedge(x);
	}

	private: fl::scalar doEvalDerivative(fl::scalar x) const
	{
		FL_THROW2(std::runtime_error, "Derivative of a hedge should never be called");
	}


	private: fl::Hedge* p_hedge_;
}; // HedgeActivationFunction


////////////////////////////////////////////////////////////////////////////////
// Neurons


class InputNode: public fl::ann::InputNeuron<fl::scalar>
{
	public: typedef fl::ann::InputNeuron<fl::scalar> BaseType;


	public: explicit InputNode(fl::Variable* p_var,
							   fl::ann::Layer<fl::scalar>* p_layer = fl::null)
	: BaseType(p_layer),
	  p_var_(p_var)
	{
	}

	public: fl::Variable* getVariable() const
	{
		return p_var_;
	}


	private: fl::Variable* p_var_;
}; // InputNode

class TermNode: public fl::ann::Neuron<fl::scalar>
{
	public: typedef fl::ann::Neuron<fl::scalar> BaseType;


	public: explicit TermNode(fl::Term* p_term,
							  fl::ann::Layer<fl::scalar>* p_layer = fl::null)
	: BaseType(new fl::ann::WeightedSumNetInputFunction(),
			   new TermActivationFunction(p_term),
			   p_layer),
	  p_term_(p_term)
	{
	}

	public: fl::Term* getTerm() const
	{
		return p_term_;
	}


	private: fl::Term* p_term_;
}; // TermNode

class HedgeNode: public fl::ann::Neuron<fl::scalar>
{
	public: typedef fl::ann::Neuron<fl::scalar> BaseType;


	public: explicit HedgeNode(fl::Hedge* p_hedge,
							   fl::ann::Layer<fl::scalar>* p_layer = fl::null)
	: BaseType(new fl::ann::WeightedSumNetInputFunction(),
			   new HedgeActivationFunction(p_hedge),
			   p_layer),
	  p_hedge_(p_hedge)
	{
	}

	public: fl::Hedge* getHedge() const
	{
		return p_hedge_;
	}


	private: fl::Hedge* p_hedge_;
}; // HedgeNode

/// A neuron that applies a norm operator to its input
class NormNode: public fl::ann::Neuron<fl::scalar>
{
	public: typedef fl::ann::Neuron<fl::scalar> BaseType;


	public: explicit NormNode(fl::Norm* p_norm,
							  fl::ann::Layer<fl::scalar>* p_layer = fl::null)
	: BaseType(new NormNetInputFunction(p_norm),
			   new fl::ann::PureLinearActivationFunction(),
			   p_layer),
	  p_norm_(p_norm)
	{
	}

	public: fl::Norm* getNorm() const
	{
		return p_norm_;
	}


	private: fl::Norm* p_norm_;
}; // NormNode

}} // Namespace fl::anfis

#endif // FL_ANFIS_MODEL_H
