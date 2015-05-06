/**
 * \file fl/detail/terms.h
 *
 * \brief Functionalities related to fuzzy terms
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

#ifndef FL_DETAIL_TERMS_H
#define FL_DETAIL_TERMS_H


#include <cstddef>
#include <fl/detail/math.h>
#include <fl/fuzzylite.h>
#include <fl/term/Term.h>
#include <fl/term/Bell.h>
#include <fl/term/Concave.h>
#include <fl/term/Constant.h>
#include <fl/term/Cosine.h>
#include <fl/term/Discrete.h>
#include <fl/term/Gaussian.h>
#include <fl/term/GaussianProduct.h>
#include <fl/term/Linear.h>
#include <fl/term/PiShape.h>
#include <fl/term/Ramp.h>
#include <fl/term/Rectangle.h>
#include <fl/term/Sigmoid.h>
#include <fl/term/SigmoidDifference.h>
#include <fl/term/SigmoidProduct.h>
#include <fl/term/SShape.h>
#include <fl/term/Triangle.h>
#include <fl/term/Trapezoid.h>
#include <fl/term/ZShape.h>
#include <vector>


namespace fl { namespace detail {

/// Returns the parameters associated to the given (pointer to) term \a p_term
std::vector<fl::scalar> GetTermParameters(const fl::Term* p_term);

/// Sets the parameters defined by the iterator range [\a first, \a last) for the given (pointer to) term \a p_term
template <typename IterT>
void SetTermParameters(fl::Term* p_term, IterT first, IterT last);

/// Evaluates the partial derivatives of the given Bell term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalBellTermDerivativeWrtParams(const fl::Bell& term, fl::scalar x);

/// Evaluates the partial derivatives of the given concave term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalConcaveTermDerivativeWrtParams(const fl::Concave& term, fl::scalar x);

/// Evaluates the partial derivatives of the given constant term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalConstantTermDerivativeWrtParams(const fl::Constant& term, fl::scalar x);

/// Evaluates the partial derivatives of the given Cosine term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalCosineTermDerivativeWrtParams(const fl::Cosine& term, fl::scalar x);

/// Evaluates the partial derivatives of the given Gaussian term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalGaussianTermDerivativeWrtParams(const fl::Gaussian& term, fl::scalar x);

/// Evaluates the partial derivatives of the given Gaussian product term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalGaussianProductTermDerivativeWrtParams(const fl::GaussianProduct& term, fl::scalar x);

/// Evaluates the partial derivatives of the given Pi-shape term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalPiShapeTermDerivativeWrtParams(const fl::PiShape& term, fl::scalar x);

/// Evaluates the partial derivatives of the given ramp term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalRampTermDerivativeWrtParams(const fl::Ramp& term, fl::scalar x);

/// Evaluates the partial derivatives of the given rectangle term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalRectangleTermDerivativeWrtParams(const fl::Rectangle& term, fl::scalar x);

/// Evaluates the partial derivatives of the given sigmoid term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalSigmoidTermDerivativeWrtParams(const fl::Sigmoid& term, fl::scalar x);

/// Evaluates the partial derivatives of the given sigmoid difference term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalSigmoidDifferenceTermDerivativeWrtParams(const fl::SigmoidDifference& term, fl::scalar x);

/// Evaluates the partial derivatives of the given sigmoid product term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalSigmoidProductTermDerivativeWrtParams(const fl::SigmoidProduct& term, fl::scalar x);

/// Evaluates the partial derivatives of the given S-shape term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalSShapeTermDerivativeWrtParams(const fl::SShape& term, fl::scalar x);

/// Evaluates the partial derivatives of the given Trapezoid term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalTrapezoidTermDerivativeWrtParams(const fl::Trapezoid& term, fl::scalar x);

/// Evaluates the partial derivatives of the given Triangle term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalTriangleTermDerivativeWrtParams(const fl::Triangle& term, fl::scalar x);

/// Evaluates the partial derivatives of the given Z-shape term \a term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalZShapeTermDerivativeWrtParams(const fl::ZShape& term, fl::scalar x);

/// Evaluates the partial derivatives of the given (pointer to a) term \a p_term for the given value \a x with respect to its parameters
std::vector<fl::scalar> EvalTermDerivativeWrtParams(const fl::Term* p_term, fl::scalar x);


////////////////////////
// Template definitions
////////////////////////


template <typename IterT>
void SetTermParameters(fl::Term* p_term, IterT first, IterT last)
{
    //FIXME: it would be a good idea to add a pure virtual method in fl::Term
    //       class that returns the vector of parameters, like:
    //         virtual std::vector<fl::scalar> getParameters() = 0;

    const std::vector<fl::scalar> params(first, last);

    if (dynamic_cast<fl::Bell*>(p_term))
    {
        fl::Bell* p_realTerm = dynamic_cast<fl::Bell*>(p_term);
        p_realTerm->setCenter(params[0]);
        p_realTerm->setWidth(params[1]);
        p_realTerm->setSlope(params[2]);
    }
    else if (dynamic_cast<fl::Concave*>(p_term))
    {
        fl::Concave* p_realTerm = dynamic_cast<fl::Concave*>(p_term);
        p_realTerm->setInflection(params[0]);
        p_realTerm->setEnd(params[1]);
    }
    else if (dynamic_cast<fl::Constant*>(p_term))
    {
        fl::Constant* p_realTerm = dynamic_cast<fl::Constant*>(p_term);
        p_realTerm->setValue(params[0]);
    }
    else if (dynamic_cast<fl::Cosine*>(p_term))
    {
        fl::Cosine* p_realTerm = dynamic_cast<fl::Cosine*>(p_term);
        p_realTerm->setCenter(params[0]);
        p_realTerm->setWidth(params[1]);
    }
    else if (dynamic_cast<fl::Discrete*>(p_term))
    {
        fl::Discrete* p_realTerm = dynamic_cast<fl::Discrete*>(p_term);
        const std::size_t np = params.size();
        std::vector<fl::Discrete::Pair> pairs;
        for (std::size_t p = 0; p < (np-1); p += 2)
        {
            pairs.push_back(fl::Discrete::Pair(params[p], params[p+1]));
        }
        p_realTerm->setXY(pairs);
    }
    else if (dynamic_cast<fl::Gaussian*>(p_term))
    {
        fl::Gaussian* p_realTerm = dynamic_cast<fl::Gaussian*>(p_term);
        p_realTerm->setMean(params[0]);
        p_realTerm->setStandardDeviation(params[1]);
    }
    else if (dynamic_cast<fl::GaussianProduct*>(p_term))
    {
        fl::GaussianProduct* p_realTerm = dynamic_cast<fl::GaussianProduct*>(p_term);
        p_realTerm->setMeanA(params[0]);
        p_realTerm->setStandardDeviationA(params[1]);
        p_realTerm->setMeanB(params[2]);
        p_realTerm->setStandardDeviationB(params[3]);
    }
    else if (dynamic_cast<fl::Linear*>(p_term))
    {
        fl::Linear* p_realTerm = dynamic_cast<fl::Linear*>(p_term);
        p_realTerm->setCoefficients(params);
    }
    else if (dynamic_cast<fl::PiShape*>(p_term))
    {
        fl::PiShape* p_realTerm = dynamic_cast<fl::PiShape*>(p_term);
        p_realTerm->setBottomLeft(params[0]);
        p_realTerm->setTopLeft(params[1]);
        p_realTerm->setTopRight(params[2]);
        p_realTerm->setBottomRight(params[3]);
    }
    else if (dynamic_cast<fl::Ramp*>(p_term))
    {
        fl::Ramp* p_realTerm = dynamic_cast<fl::Ramp*>(p_term);
        p_realTerm->setStart(params[0]);
        p_realTerm->setEnd(params[1]);
    }
    else if (dynamic_cast<fl::Rectangle*>(p_term))
    {
        fl::Rectangle* p_realTerm = dynamic_cast<fl::Rectangle*>(p_term);
        p_realTerm->setStart(params[0]);
        p_realTerm->setEnd(params[1]);
    }
    else if (dynamic_cast<fl::Sigmoid*>(p_term))
    {
        fl::Sigmoid* p_realTerm = dynamic_cast<fl::Sigmoid*>(p_term);
        p_realTerm->setInflection(params[0]);
        p_realTerm->setSlope(params[1]);
    }
    else if (dynamic_cast<fl::SigmoidDifference*>(p_term))
    {
        fl::SigmoidDifference* p_realTerm = dynamic_cast<fl::SigmoidDifference*>(p_term);
        p_realTerm->setLeft(params[0]);
        p_realTerm->setRising(params[1]);
        p_realTerm->setFalling(params[2]);
        p_realTerm->setRight(params[3]);
    }
    else if (dynamic_cast<fl::SigmoidProduct*>(p_term))
    {
        fl::SigmoidProduct* p_realTerm = dynamic_cast<fl::SigmoidProduct*>(p_term);
        p_realTerm->setLeft(params[0]);
        p_realTerm->setRising(params[1]);
        p_realTerm->setFalling(params[2]);
        p_realTerm->setRight(params[3]);
    }
    else if (dynamic_cast<fl::SShape*>(p_term))
    {
        fl::SShape* p_realTerm = dynamic_cast<fl::SShape*>(p_term);
        p_realTerm->setStart(params[0]);
        p_realTerm->setEnd(params[1]);
    }
    else if (dynamic_cast<fl::Trapezoid*>(p_term))
    {
        fl::Trapezoid* p_realTerm = dynamic_cast<fl::Trapezoid*>(p_term);
        p_realTerm->setVertexA(params[0]);
        p_realTerm->setVertexB(params[1]);
        p_realTerm->setVertexC(params[2]);
        p_realTerm->setVertexD(params[3]);
    }
    else if (dynamic_cast<fl::Triangle*>(p_term))
    {
        fl::Triangle* p_realTerm = dynamic_cast<fl::Triangle*>(p_term);
        p_realTerm->setVertexA(params[0]);
        p_realTerm->setVertexB(params[1]);
        p_realTerm->setVertexC(params[2]);
    }
    else if (dynamic_cast<fl::ZShape*>(p_term))
    {
        fl::ZShape* p_realTerm = dynamic_cast<fl::ZShape*>(p_term);
        p_realTerm->setStart(params[0]);
        p_realTerm->setEnd(params[1]);
    }
}

}} // Namespace fl::detail


#endif // FL_DETAIL_TERMS_H
