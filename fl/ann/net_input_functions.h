/**
 * \file fl/ann/net_input_functions.h
 *
 * \brief Net input functions for a neuron
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

#ifndef Fl_ANN_NET_INPUT_FUNCTIONS_H
#define Fl_ANN_NET_INPUT_FUNCTIONS_H


#include <boost/noncopyable.hpp>
#include <iterator>
#include <my/commons.h>
#include <my/detail/iterators.h>


namespace fl { namespace ann {

template <typename ValueT>
class NetInputFunction: boost::noncopyable
{
	protected: typedef FL_ForwardIteratorType(ValueT) ForwardIterator;


	public: virtual ~NetInputFunction() { }

	public: template <typename InputIterT, typename WeightIterT>
			ValueT eval(InputIterT inputFirst, InputIterT inputLast,
						WeightIterT weightFirst, WeightIterT weightLast) const
	{
		return this->doEval(inputFirst, inputLast, weightFirst, weightLast);
	}

	public: template <typename InputIterT, typename WeightIterT>
			std::vector<ValueT> evalDerivativeWrtInput(InputIterT inputFirst, InputIterT inputLast,
													   WeightIterT weightFirst, WeightIterT weightLast) const
	{
		return this->doEvalDerivativeWrtInput(inputFirst, inputLast, weightFirst, weightLast);
	}

	public: template <typename InputIterT, typename WeightIterT>
			std::vector<ValueT> evalDerivativeWrtWeight(InputIterT inputFirst, InputIterT inputLast,
													    WeightIterT weightFirst, WeightIterT weightLast) const
	{
		return this->doEvalDerivativeWrtWeight(inputFirst, inputLast, weightFirst, weightLast);
	}

	private: virtual ValueT doEval(ForwardIterator inputFirst, ForwardIterator inputLast,
								   ForwardIterator weightFirst, ForwardIterator weightLast) const = 0;

	private: virtual std::vector<ValueT> doEvalDerivativeWrtInput(ForwardIterator inputFirst, ForwardIterator inputLast,
																  ForwardIterator weightFirst, ForwardIterator weightLast) const = 0;

	private: virtual std::vector<ValueT> doEvalDerivativeWrtWeight(ForwardIterator inputFirst, ForwardIterator inputLast,
																   ForwardIterator weightFirst, ForwardIterator weightLast) const = 0;
}; // NetInputFunction


template <typename ValueT>
class ConstantNetInputFunction: public NetInputFunction<ValueT>
{
	private: typedef NetInputFunction<ValueT> BaseT;
	private: typedef typename BaseT::ForwardIterator ForwardIterator;


	public: ConstantNetInputFunction(ValueT c)
	: c_(c)
	{
	}

	private: ValueT doEval(ForwardIterator inputFirst, ForwardIterator inputLast,
						   ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( inputFirst );
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( inputLast );
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( weightFirst );
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( weightLast );

		return c_;
	}

	private: std::vector<ValueT> doEvalDerivativeWrtInput(ForwardIterator inputFirst, ForwardIterator inputLast,
														  ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( weightFirst );
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( weightLast );

		const std::ptrdiff_t n = std::distance(inputFirst, inputLast);
		return std::vector<ValueT>(n, 0);
		//std::vector<ValueT> v(inputFirst, inputLast);
		//return std::vector<ValueT>(v.size(), 0);
	}

	private: std::vector<ValueT> doEvalDerivativeWrtWeight(ForwardIterator inputFirst, ForwardIterator inputLast,
														   ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( inputFirst );
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( inputLast );

		const std::ptrdiff_t n = std::distance(weightFirst, weightLast);
		return std::vector<ValueT>(n, 0);
		//std::vector<ValueT> v(weightFirst, weightLast);
		//return std::vector<ValueT>(v.size(), 0);
	}


	private: const ValueT c_;
}; // ConstantNetInputFunction


template <typename ValueT>
class WeightedSumNetInputFunction: public NetInputFunction<ValueT>
{
	private: typedef NetInputFunction<ValueT> BaseT;
	private: typedef typename BaseT::ForwardIterator ForwardIterator;


	private: ValueT doEval(ForwardIterator inputFirst, ForwardIterator inputLast,
						   ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		ValueT res = 0;

		while (inputFirst != inputLast && weightFirst != weightLast)
		{
			res += *inputFirst * *weightFirst;

			++inputFirst;
			++weightFirst;
		}

		return res;
	}

	private: std::vector<ValueT> doEvalDerivativeWrtInput(ForwardIterator inputFirst, ForwardIterator inputLast,
														  ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( inputFirst );
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( inputLast );

		return std::vector<ValueT>(weightFirst, weightLast);
	}

	private: std::vector<ValueT> doEvalDerivativeWrtWeight(ForwardIterator inputFirst, ForwardIterator inputLast,
														   ForwardIterator weightFirst, ForwardIterator weightLast) const
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( weightFirst );
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( weightLast );

		return std::vector<ValueT>(inputFirst, inputLast);
	}
}; // WeightedSumNetInputFunction

}} // Namespace fl::ann

#endif // Fl_ANN_NET_INPUT_FUNCTIONS_H
