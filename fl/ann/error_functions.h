/**
 * \file fl/ann/error_functions.h
 *
 * \brief Error functions for learning algorithms
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

#ifndef FL_ANN_ERROR_FUNCTIONS_H
#define FL_ANN_ERROR_FUNCTIONS_H


#include <boost/noncopyable.hpp>
#include <my/detail/iterators.h>
#include <vector>


namespace fl { namespace ann {

/**
 * The error function that is to be minimized.
 *
 * The <em>error function</em> (also known as <em>performance function</em>) is
 * the function that is to be minimized.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class ErrorFunction: boost::noncopyable
{
    protected: typedef FL_ForwardIteratorType(ValueT) ForwardIterator;


    public: ErrorFunction()
    {
    }

    public: virtual ~ErrorFunction() { }

    public: void reset()
    {
        this->doReset(); 
    }

    public: template <typename TargetIterT, typename ActualIterT>
            void update(TargetIterT targetFirst, TargetIterT targetLast,
                        ActualIterT actualFirst, ActualIterT actualLast)
    {
		this->doUpdate(targetFirst, targetLast, actualFirst, actualLast);
	}

    public: ValueT getTotalError() const
    {
        return this->doGetTotalError();
    }

    /// Returns E(t,o), where t is the target output and o is the actual output
    public: template <typename TargetIterT, typename ActualIterT>
            ValueT eval(TargetIterT targetFirst, TargetIterT targetLast,
                        ActualIterT actualFirst, ActualIterT actualLast)
    {
        return this->doEval(targetFirst, targetLast, actualFirst, actualLast);
    }

    /// Returns dE(t,o)/do_j, where t is the target output and o is the actual output
    public: template <typename TargetIterT, typename ActualIterT>
            std::vector<ValueT> evalDerivativeWrtOutput(TargetIterT targetFirst, TargetIterT targetLast,
                                                        ActualIterT actualFirst, ActualIterT actualLast)
    {
        return this->doEvalDerivativeWrtOutput(targetFirst, targetLast, actualFirst, actualLast);
    }

    private: virtual void doReset() = 0;

    private: virtual void doUpdate(ForwardIterator targetFirst, ForwardIterator targetLast,
                                   ForwardIterator actualFirst, ForwardIterator actualLast) = 0;

    private: virtual ValueT doGetTotalError() const = 0;

    private: virtual ValueT doEval(ForwardIterator targetFirst, ForwardIterator targetLast,
                                   ForwardIterator actualFirst, ForwardIterator actualLast) = 0;

    private: virtual std::vector<ValueT> doEvalDerivativeWrtOutput(ForwardIterator targetFirst, ForwardIterator targetLast,
                                                                   ForwardIterator actualFirst, ForwardIterator actualLast) = 0;
}; // ErrorFunction


/**
 * Sum of Squared Error (SSE) function
 *
 * The SSE function is defined as
 * \f[
 *  E_d = \sum_k (t_{kd}-o_{kd})^2
 * \f]
 * where:
 * - \f$t_{kd}\f$ is the target output associated to the \f$k\f$-th output unit and \f$d\f$-th training pattern
 * - \f$o_{kd}\f$ is the actual output associated to the \f$k\f$-th output unit and \f$d\f$-th training pattern
 * .
 * Note, sometimes the above equation is multiplied by a factor of 0.5.
 * This is done only for convenience in calculating derivates.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class SumSquaredErrorFunction: public ErrorFunction<ValueT>
{
    private: typedef ErrorFunction<ValueT> BaseType;
    private: typedef typename BaseType::ForwardIterator ForwardIterator;


	public: explicit SumSquaredErrorFunction(ValueT factor=1)
	: factor_(factor),
	  totErr_(0)
	{
	}

	private: void doReset()
	{
		totErr_ = 0;
	}

    private: void doUpdate(ForwardIterator targetFirst, ForwardIterator targetLast,
                           ForwardIterator actualFirst, ForwardIterator actualLast)
	{
        totErr_ += this->eval(targetFirst, targetLast, actualFirst, actualLast);
	}

	private: ValueT doGetTotalError() const
	{
		return totErr_;
	}

    private: ValueT doEval(ForwardIterator targetFirst, ForwardIterator targetLast,
                           ForwardIterator actualFirst, ForwardIterator actualLast)
    {
        ValueT res = 0;

        while (targetFirst != targetLast && actualFirst != actualLast)
        {
            const ValueT e = *actualFirst - *targetFirst;

            res += e*e;

            ++targetFirst;
            ++actualFirst;
        }

        return factor_*res;
    }

    private: std::vector<ValueT> doEvalDerivativeWrtOutput(ForwardIterator targetFirst, ForwardIterator targetLast,
                                                           ForwardIterator actualFirst, ForwardIterator actualLast)
    {
        std::vector<ValueT> res;

        while (targetFirst != targetLast && actualFirst != actualLast)
        {
            const ValueT de = factor_ * 2.0 * (*actualFirst - *targetFirst);

            res.push_back(de);

            ++targetFirst;
            ++actualFirst;
        }

        return res;
    }


	private: ValueT factor_;
	private: ValueT totErr_;
}; /// SumSquaredErrorFunction

/**
 * Mean of Squared Error (MSE) function
 *
 * The MSE function is defined as
 * \f[
 *  E_d = \frac{1}{N} \sum_k (t_{kd}-o_{kd})^2
 * \f]
 * where:
 * - \f$t_{kd}\f$ is the target output associated to the \f$k\f$-th output unit and \f$d\f$-th training pattern
 * - \f$o_{kd}\f$ is the actual output associated to the \f$k\f$-th output unit and \f$d\f$-th training pattern
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class MeanSquaredErrorFunction: public ErrorFunction<ValueT>
{
    private: typedef ErrorFunction<ValueT> BaseType;
    private: typedef typename BaseType::ForwardIterator ForwardIterator;


	public: MeanSquaredErrorFunction()
	: totErr_(0),
	  count_(0)
	{
	}

	private: void doReset()
	{
		totErr_ = 0;
		count_ = 0;
	}

    private: void doUpdate(ForwardIterator targetFirst, ForwardIterator targetLast,
                           ForwardIterator actualFirst, ForwardIterator actualLast)
	{
        totErr_ += this->eval(targetFirst, targetLast, actualFirst, actualLast);
		++count_;
	}

	private: ValueT doGetTotalError() const
	{
		return count_ > 0 ? totErr_/count_ : 0;
	}

    private: ValueT doEval(ForwardIterator targetFirst, ForwardIterator targetLast,
                           ForwardIterator actualFirst, ForwardIterator actualLast)
    {
        ValueT res = 0;
        while (targetFirst != targetLast && actualFirst != actualLast)
        {
            const ValueT e = *actualFirst - *targetFirst;

            res += e*e;

            ++targetFirst;
            ++actualFirst;
        }

        return res;
    }

    private: std::vector<ValueT> doEvalDerivativeWrtOutput(ForwardIterator targetFirst, ForwardIterator targetLast,
                                                           ForwardIterator actualFirst, ForwardIterator actualLast)
    {
        std::vector<ValueT> res;

        while (targetFirst != targetLast && actualFirst != actualLast)
        {
            const ValueT de = 2.0 * (*actualFirst - *targetFirst);

            res.push_back(de);

            ++targetFirst;
            ++actualFirst;
        }

        return res;
    }


	private: ValueT totErr_;
	private: std::size_t count_;
}; // MeanSquaredErrorFunction

}} // Namespace fl::ann

#endif // FL_ANN_ERROR_FUNCTIONS_H
