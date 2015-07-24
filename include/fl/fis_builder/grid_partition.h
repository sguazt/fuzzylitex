/**
 * \file fl/fis_builder/grid_partition.h
 *
 * \brief Grid-based input partitioning method for the fuzzy identification
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2015 Marco Guazzone (marco.guazzone@gmail.com)
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

#ifndef FL_FIS_BUILDER_GRID_PARTITION_H
#define FL_FIS_BUILDER_GRID_PARTITION_H


#include <cstddef>
#include <fl/activation/General.h>
#include <fl/dataset.h>
#include <fl/defuzzifier/WeightedAverage.h>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/fuzzylite.h>
#include <fl/macro.h>
#include <fl/norm/s/Maximum.h>
#include <fl/norm/t/AlgebraicProduct.h>
#include <fl/rule/Rule.h>
#include <fl/rule/RuleBlock.h>
#include <fl/term/Accumulated.h> //FIXME: needed even if not explicitly used because of fwd decl in fl::OutputVariable
#include <fl/term/Bell.h>
#include <fl/term/Constant.h>
#include <fl/term/Gaussian.h>
#include <fl/term/GaussianProduct.h>
#include <fl/term/Linear.h>
#include <fl/term/PiShape.h>
#include <fl/term/SigmoidDifference.h>
#include <fl/term/SigmoidProduct.h>
#include <fl/term/Triangle.h>
#include <fl/term/Trapezoid.h>
#include <fl/variable/InputVariable.h>
#include <fl/variable/OutputVariable.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace fl {

/**
 * FIS builder based on the <em>grid partitioning</em> method [1].
 *
 * With the <em>grid partitioning</em> method, the multidimensional input-output
 * data space is divided into fixed partitions.
 *
 * Despite its simplicity, this method comes with several disadvantages:
 * - This method suffers of the <em>curse of dimensionality</em> problem,
 *   whereby the number of generated rules grows exponentially with the number
 *   of input variables, of input terms, of output variables, and of output
 *   terms [1,2]. Specifically, with \f$n\f$ inputs and \f$k\f$ membership
 *   functions for each input, the total number of fuzzy rules generated is
 *   \f$k^n\f$. 
 * - Also, some of the generated rules may turn out to be meaningless [2].
 * .
 *
 * References:
 * -# F.O. Karray et al., "Soft Computing and Intelligent Systems Design: Theory, Tools, and Applications,"
 * -# J.-S. R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename EngineT>
class GridPartitionFisBuilder
{
private:
    static const std::size_t DefaultNumberOfInputTerms;
    static const std::string DefaultInputTerm;
    static const std::size_t DefaultNumberOfOutputTerms;
    static const std::string DefaultOutputTerm;


public:
    GridPartitionFisBuilder();

    GridPartitionFisBuilder(std::size_t numInputTerms,
                            const std::string& inputTermClass,
                            const std::string& outputTermClass);

    template <typename NumInTermsIterT,
              typename InTermClassIterT>
    GridPartitionFisBuilder(NumInTermsIterT numInTermsFirst, NumInTermsIterT numInTermsLast,
                            InTermClassIterT inTermClassFirst, InTermClassIterT inTermClassLast,
                            const std::string& outTermClass);

    FL_unique_ptr<EngineT> build(const fl::DataSet<fl::scalar>& data);

    template <typename MatrixT>
    FL_unique_ptr<EngineT> build(const MatrixT& data, std::size_t numInputs, std::size_t numOutputs);


private:
    std::vector<std::size_t> numInTerms_;
    std::vector<std::string> inTerms_;
    std::vector<std::size_t> numOutTerms_;
    std::vector<std::string> outTerms_;
}; // GridPartitionFisBuilder


/////////////////////////
// Template definitiions
/////////////////////////


template <typename EngineT>
GridPartitionFisBuilder<EngineT>::GridPartitionFisBuilder()
: numInTerms_(1, DefaultNumberOfInputTerms),
  inTerms_(1, DefaultInputTerm),
  numOutTerms_(1, DefaultNumberOfOutputTerms),
  outTerms_(1, DefaultOutputTerm)
{
}

template <typename EngineT>
GridPartitionFisBuilder<EngineT>::GridPartitionFisBuilder(std::size_t numInputTerms,
                                                          const std::string& inputTermClass,
                                                          const std::string& outputTermClass)
: numInTerms_(1, numInputTerms),
  inTerms_(1, inputTermClass),
  numOutTerms_(1, DefaultNumberOfOutputTerms),
  outTerms_(1, outputTermClass)
{
}

template <typename EngineT>
template <typename NumInTermsIterT,
          typename InTermClassIterT>
GridPartitionFisBuilder<EngineT>::GridPartitionFisBuilder(NumInTermsIterT numInTermsFirst, NumInTermsIterT numInTermsLast,
                                                          InTermClassIterT inTermClassFirst, InTermClassIterT inTermClassLast,
                                                          const std::string& outTermClass)
: numInTerms_(numInTermsFirst, numInTermsLast),
  inTerms_(inTermClassFirst, inTermClassLast),
  numOutTerms_(1, DefaultNumberOfOutputTerms),
  outTerms_(1, outTermClass)
{
}

/*
template <typename EngineT>
FL_unique_ptr<EngineT> GridPartitionFisBuilder<EngineT>::build(const fl::DataSet<fl::scalar>& data)
{
    const std::size_t numInputs =  data.numOfInputs();
    const std::size_t numOutputs =  data.numOfOutputs();
    const std::size_t numInOuts = numInputs+numOutputs;

    //if (numOutputs > 1)
    //{
    //  FL_THROW2(std::invalid_argument, "Data must have exactly one output");
    //}
    if (outTerms_.front() != fl::Linear().className() && outTerms_.front() != fl::Constant().className())
    {
        FL_THROW2(std::invalid_argument, "Output term must be either linear or constant");
    }

    // Adjust internal state
    if (numInputs != numInTerms_.size())
    {
        numInTerms_.resize(numInputs, DefaultNumberOfInputTerms);
        inTerms_.resize(numInputs, DefaultInputTerm);
    }
    if (numOutputs != numOutTerms_.size())
    {
        numOutTerms_.resize(numInputs, DefaultNumberOfOutputTerms);
        outTerms_.resize(numOutputs, DefaultOutputTerm);
    }
    // Compute data range
    std::vector<fl::scalar> mins(numInOuts, fl::inf);
    std::vector<fl::scalar> maxs(numInOuts, -fl::inf);
    for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
                                                              entryEndIt = data.entryEnd();
         entryIt != entryEndIt;
         ++entryIt)
    {
        for (std::size_t i = 0; i < numInOuts; ++i)
        {
            mins[i] = std::min(entryIt->getField(i), mins[i]);
            maxs[i] = std::max(entryIt->getField(i), maxs[i]);
        }
    }
    // Fix for zero range
    for (std::size_t i = 0; i < numInOuts; ++i)
    {
        if (fl::detail::FloatTraits<fl::scalar>::ApproximatelyZero(maxs[i], mins[i]))
        {
            const fl::scalar delta = 0.1*std::abs(maxs[i]-mins[i])+0.1;

            mins[i] -= delta;
            maxs[i] += delta;
        }
    }

    FL_unique_ptr<EngineT> p_fis(new EngineT());

    std::size_t numRules = 1;

    // Generate input variables and terms
    std::vector<fl::InputVariable*> inputs(numInputs);
    for (std::size_t i = 0; i < numInputs; ++i)
    {
        std::ostringstream oss;

        fl::InputVariable* p_iv = new fl::InputVariable();

        oss << "in" << i;
        p_iv->setEnabled(true);
        p_iv->setName(oss.str());
        p_iv->setRange(mins[i], maxs[i]);

        const std::size_t numTerms = numInTerms_[i];
        const std::string termClass = inTerms_[i];

        for (std::size_t j = 0; j < numTerms; ++j)
        {
            // Give a name to this term
            oss.str("");
            oss << p_iv->getName() << "mf" << j;

            if (termClass == fl::Bell().className())
            {
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar width = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar slope = 2;

                p_iv->addTerm(new fl::Bell(oss.str(), center, width, slope));
            }
            else if (termClass == fl::Gaussian().className())
            {
                const fl::scalar mean = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar sd = (0.5*(maxs[i]-mins[i])/(numTerms-1))/std::sqrt(2.0*std::log(2));

                p_iv->addTerm(new fl::Gaussian(oss.str(), mean, sd));
            }
            else if (termClass == fl::GaussianProduct().className())
            {
                const fl::scalar sd0 = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar mean = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar mean1 = mean - 0.6*sd0;
                const fl::scalar mean2 = mean + 0.6*sd0;
                const fl::scalar sd = 0.4*sd0/std::sqrt(2*std::log(2));

                p_iv->addTerm(new fl::GaussianProduct(oss.str(), mean1, sd, mean2, sd));
            }
            else if (termClass == fl::SigmoidDifference().className())
            {
                const fl::scalar range = maxs[i]-mins[i];
                const fl::scalar offs = 0.5*range/(numTerms-1);
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar slope = 10.0/range*numTerms;
                const fl::scalar left = center-offs;
                const fl::scalar rising = slope;
                const fl::scalar falling = slope;
                const fl::scalar right = center+offs;

                p_iv->addTerm(new fl::SigmoidDifference(oss.str(), left, rising, falling, right));
            }
            else if (termClass == fl::SigmoidProduct().className())
            {
                const fl::scalar range = maxs[i]-mins[i];
                const fl::scalar offs = 0.5*range/(numTerms-1);
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar slope = 10.0/range*numTerms;
                const fl::scalar left = center-offs;
                const fl::scalar rising = slope;
                const fl::scalar falling = -slope;
                const fl::scalar right = center+offs;

                p_iv->addTerm(new fl::SigmoidProduct(oss.str(), left, rising, falling, right));
            }
            else if (termClass == fl::Triangle().className())
            {
                const fl::scalar b = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar offs = (maxs[i]-mins[i])/static_cast<fl::scalar>(numTerms-1);
                const fl::scalar a = b-offs;
                const fl::scalar c = b+offs;

                p_iv->addTerm(new fl::Triangle(oss.str(), a, b, c));
            }
            else if (termClass == fl::Trapezoid().className())
            {
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar offs = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar a = center-1.4*offs;
                const fl::scalar b = center-0.6*offs;
                const fl::scalar c = center+0.6*offs;
                const fl::scalar d = center+1.4*offs;

                p_iv->addTerm(new fl::Trapezoid(oss.str(), a, b, c, d));
            }
            else if (termClass == fl::PiShape().className())
            {
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar offs = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar bottomLeft = center-1.4*offs;
                const fl::scalar topLeft = center-0.6*offs;
                const fl::scalar topRight = center+0.6*offs;
                const fl::scalar bottomRight = center+1.4*offs;

                p_iv->addTerm(new fl::PiShape(oss.str(), bottomLeft, topLeft, topRight, bottomRight));
            }
            else
            {
                oss.str("");
                oss << "Implementation for term '" << termClass << "' is not available yet";
                FL_THROW2(std::runtime_error, oss.str());
            }
        }

        p_fis->addInputVariable(p_iv);

        numRules *= numTerms;
    }

    // Generate output variables and terms
    std::vector<fl::OutputVariable*> outputs(numOutputs);
    for (std::size_t i = 0; i < numOutputs; ++i)
    {
        const std::size_t k = numInputs+i;

        fl::OutputVariable* p_ov = new fl::OutputVariable();

        std::ostringstream oss;

        oss << "out" << i;
        p_ov->setEnabled(true);
        p_ov->setName(oss.str());
        p_ov->setRange(mins[k], maxs[k]);
        //p_ov->fuzzyOutput()->setAccumulation(new fl::Maximum());
        p_ov->fuzzyOutput()->setAccumulation(fl::null);
        p_ov->setDefuzzifier(new fl::WeightedAverage());
        p_ov->setDefaultValue(fl::nan);
        p_ov->setPreviousValue(false);

        for (std::size_t j = 0; j < numRules; ++j)
        {
            // Give a name to this term
            oss.str("");
            oss << p_ov->getName() << "mf" << j;

            if (outTerms_[i] == fl::Constant().className())
            {
                p_ov->addTerm(new fl::Constant(oss.str(), 0));
            }
            else if (outTerms_[i] == fl::Linear().className())
            {
                p_ov->addTerm(new fl::Linear(oss.str(), std::vector<fl::scalar>(numInputs+1, 0), p_fis.get()));
            }
        }

        p_fis->addOutputVariable(p_ov);
    }

    // Generate rules
    fl::RuleBlock* p_rules = new fl::RuleBlock();
    p_rules->setEnabled(true);
    p_rules->setConjunction(new fl::AlgebraicProduct());
    p_rules->setDisjunction(new fl::Maximum());
    //p_rules->setActivation(new fl::AlgebraicProduct());
    p_rules->setActivation(fl::null);
    for (std::size_t r = 0; r < numRules; ++r)
    {
        std::ostringstream oss;

        oss << fl::Rule::ifKeyword() << " ";

        std::size_t tmp = r;
#ifdef FL_DEBUG
        // Generates rule in the same order of MATLAB
        std::vector<std::size_t> ruleTerms(numInputs);
        for (std::size_t j = numInputs; j > 0; --j)
        {
            std::size_t jj = j-1;
            const std::size_t termIdx = tmp % numInTerms_[jj];

            ruleTerms[jj] = termIdx;

            tmp = static_cast<std::size_t>(std::floor(tmp/static_cast<double>(numInTerms_[jj])));
        }
        for (std::size_t j = 0; j < numInputs; ++j)
        {
            const fl::InputVariable* p_iv = p_fis->getInputVariable(j);

            oss << p_iv->getName() << " " << fl::Rule::isKeyword() << " " << p_iv->getTerm(ruleTerms[j])->getName() << " ";

            if (j < (numInputs-1))
            {
                oss << fl::Rule::andKeyword() << " ";
            }
        }
#else // FL_DEBUG
        for (std::size_t j = 0; j < numInputs; ++j)
        {
            const fl::InputVariable* p_iv = p_fis->getInputVariable(j);
            const std::size_t termIdx = tmp % numInTerms_[j];

            oss << p_iv->getName() << " " << fl::Rule::isKeyword() << " " << p_iv->getTerm(termIdx)->getName() << " ";

            if (j < (numInputs-1))
            {
                oss << fl::Rule::andKeyword() << " ";
            }

            tmp = static_cast<std::size_t>(std::floor(tmp/static_cast<double>(numInTerms_[j])));
        }
#endif // FL_DEBUG

        oss << fl::Rule::thenKeyword();

        for (std::size_t j = 0; j < numOutputs; ++j)
        {
            const fl::OutputVariable* p_ov = p_fis->getOutputVariable(j);
            oss << " " << p_ov->getName() << " " << fl::Rule::isKeyword() << " " << p_ov->getTerm(r)->getName();
            if (j < (numOutputs-1))
            {
                oss << " " << fl::Rule::andKeyword() << " ";
            }
        }
        p_rules->addRule(fl::Rule::parse(oss.str(), p_fis.get()));
    }
    p_fis->addRuleBlock(p_rules);

    return p_fis;
}
*/

template <typename EngineT>
FL_unique_ptr<EngineT> GridPartitionFisBuilder<EngineT>::build(const fl::DataSet<fl::scalar>& data)
{
    return this->build(data.data(), data.numOfInputs(), data.numOfOutputs());
}

template <typename EngineT>
template <typename MatrixT>
FL_unique_ptr<EngineT> GridPartitionFisBuilder<EngineT>::build(const MatrixT& data, std::size_t numInputs, std::size_t numOutputs)
{
    const std::size_t numInOuts = numInputs+numOutputs;
    const std::size_t numData = data.size();

    //if (numOutputs > 1)
    //{
    //  FL_THROW2(std::invalid_argument, "Data must have exactly one output");
    //}
    if (outTerms_.front() != fl::Linear().className() && outTerms_.front() != fl::Constant().className())
    {
        FL_THROW2(std::invalid_argument, "Output term must be either linear or constant");
    }

    // Adjust internal state
    if (numInputs != numInTerms_.size())
    {
        numInTerms_.resize(numInputs, DefaultNumberOfInputTerms);
        inTerms_.resize(numInputs, DefaultInputTerm);
    }
    if (numOutputs != numOutTerms_.size())
    {
        numOutTerms_.resize(numInputs, DefaultNumberOfOutputTerms);
        outTerms_.resize(numOutputs, DefaultOutputTerm);
    }
    // Compute data range
    std::vector<fl::scalar> mins(numInOuts, fl::inf);
    std::vector<fl::scalar> maxs(numInOuts, -fl::inf);
    for (std::size_t k = 0; k < numData; ++k)
    {
        for (std::size_t i = 0; i < numInOuts; ++i)
        {
            mins[i] = std::min(data[k][i], mins[i]);
            maxs[i] = std::max(data[k][i], maxs[i]);
        }
    }
    // Fix for zero range
    for (std::size_t i = 0; i < numInOuts; ++i)
    {
        if (fl::detail::FloatTraits<fl::scalar>::ApproximatelyZero(maxs[i], mins[i]))
        {
            const fl::scalar delta = 0.1*std::abs(maxs[i]-mins[i])+0.1;

            mins[i] -= delta;
            maxs[i] += delta;
        }
    }

    FL_unique_ptr<EngineT> p_fis(new EngineT());

    std::size_t numRules = 1;

    // Generate input variables and terms
    std::vector<fl::InputVariable*> inputs(numInputs);
    for (std::size_t i = 0; i < numInputs; ++i)
    {
        std::ostringstream oss;

        fl::InputVariable* p_iv = new fl::InputVariable();

        oss << "in" << i;
        p_iv->setEnabled(true);
        p_iv->setName(oss.str());
        p_iv->setRange(mins[i], maxs[i]);

        const std::size_t numTerms = numInTerms_[i];
        const std::string termClass = inTerms_[i];

        for (std::size_t j = 0; j < numTerms; ++j)
        {
            // Give a name to this term
            oss.str("");
            oss << p_iv->getName() << "mf" << j;

            if (termClass == fl::Bell().className())
            {
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar width = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar slope = 2;

                p_iv->addTerm(new fl::Bell(oss.str(), center, width, slope));
            }
            else if (termClass == fl::Gaussian().className())
            {
                const fl::scalar mean = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar sd = (0.5*(maxs[i]-mins[i])/(numTerms-1))/std::sqrt(2.0*std::log(2));

                p_iv->addTerm(new fl::Gaussian(oss.str(), mean, sd));
            }
            else if (termClass == fl::GaussianProduct().className())
            {
                const fl::scalar sd0 = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar mean = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar mean1 = mean - 0.6*sd0;
                const fl::scalar mean2 = mean + 0.6*sd0;
                const fl::scalar sd = 0.4*sd0/std::sqrt(2*std::log(2));

                p_iv->addTerm(new fl::GaussianProduct(oss.str(), mean1, sd, mean2, sd));
            }
            else if (termClass == fl::SigmoidDifference().className())
            {
                const fl::scalar range = maxs[i]-mins[i];
                const fl::scalar offs = 0.5*range/(numTerms-1);
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar slope = 10.0/range*numTerms;
                const fl::scalar left = center-offs;
                const fl::scalar rising = slope;
                const fl::scalar falling = slope;
                const fl::scalar right = center+offs;

                p_iv->addTerm(new fl::SigmoidDifference(oss.str(), left, rising, falling, right));
            }
            else if (termClass == fl::SigmoidProduct().className())
            {
                const fl::scalar range = maxs[i]-mins[i];
                const fl::scalar offs = 0.5*range/(numTerms-1);
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar slope = 10.0/range*numTerms;
                const fl::scalar left = center-offs;
                const fl::scalar rising = slope;
                const fl::scalar falling = -slope;
                const fl::scalar right = center+offs;

                p_iv->addTerm(new fl::SigmoidProduct(oss.str(), left, rising, falling, right));
            }
            else if (termClass == fl::Triangle().className())
            {
                const fl::scalar b = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar offs = (maxs[i]-mins[i])/static_cast<fl::scalar>(numTerms-1);
                const fl::scalar a = b-offs;
                const fl::scalar c = b+offs;

                p_iv->addTerm(new fl::Triangle(oss.str(), a, b, c));
            }
            else if (termClass == fl::Trapezoid().className())
            {
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar offs = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar a = center-1.4*offs;
                const fl::scalar b = center-0.6*offs;
                const fl::scalar c = center+0.6*offs;
                const fl::scalar d = center+1.4*offs;

                p_iv->addTerm(new fl::Trapezoid(oss.str(), a, b, c, d));
            }
            else if (termClass == fl::PiShape().className())
            {
                const fl::scalar center = fl::detail::LinSpace(mins[i], maxs[i], numTerms)[j];
                const fl::scalar offs = 0.5*(maxs[i]-mins[i])/(numTerms-1);
                const fl::scalar bottomLeft = center-1.4*offs;
                const fl::scalar topLeft = center-0.6*offs;
                const fl::scalar topRight = center+0.6*offs;
                const fl::scalar bottomRight = center+1.4*offs;

                p_iv->addTerm(new fl::PiShape(oss.str(), bottomLeft, topLeft, topRight, bottomRight));
            }
            else
            {
                oss.str("");
                oss << "Implementation for term '" << termClass << "' is not available yet";
                FL_THROW2(std::runtime_error, oss.str());
            }
        }

        p_fis->addInputVariable(p_iv);

        numRules *= numTerms;
    }

    // Generate output variables and terms
    std::vector<fl::OutputVariable*> outputs(numOutputs);
    for (std::size_t i = 0; i < numOutputs; ++i)
    {
        const std::size_t k = numInputs+i;

        fl::OutputVariable* p_ov = new fl::OutputVariable();

        std::ostringstream oss;

        oss << "out" << i;
        p_ov->setEnabled(true);
        p_ov->setName(oss.str());
        p_ov->setRange(mins[k], maxs[k]);
        //p_ov->fuzzyOutput()->setAccumulation(new fl::Maximum());
        p_ov->fuzzyOutput()->setAccumulation(fl::null);
        p_ov->setDefuzzifier(new fl::WeightedAverage());
        p_ov->setDefaultValue(fl::nan);
        p_ov->setPreviousValue(false);

        for (std::size_t j = 0; j < numRules; ++j)
        {
            // Give a name to this term
            oss.str("");
            oss << p_ov->getName() << "mf" << j;

            if (outTerms_[i] == fl::Constant().className())
            {
                p_ov->addTerm(new fl::Constant(oss.str(), 0));
            }
            else if (outTerms_[i] == fl::Linear().className())
            {
                p_ov->addTerm(new fl::Linear(oss.str(), std::vector<fl::scalar>(numInputs+1, 0), p_fis.get()));
            }
        }

        p_fis->addOutputVariable(p_ov);
    }

    // Generate rules
    fl::RuleBlock* p_rules = new fl::RuleBlock();
    p_rules->setEnabled(true);
    p_rules->setConjunction(new fl::AlgebraicProduct());
    p_rules->setDisjunction(new fl::Maximum());
    ////p_rules->setActivation(new fl::AlgebraicProduct());
    //p_rules->setActivation(fl::null);
    p_rules->setActivation(new fl::General());
    p_rules->setImplication(new fl::AlgebraicProduct());
    for (std::size_t r = 0; r < numRules; ++r)
    {
        std::ostringstream oss;

        oss << fl::Rule::ifKeyword() << " ";

        std::size_t tmp = r;
#ifdef FL_DEBUG
        // Generates rule in the same order of MATLAB
        std::vector<std::size_t> ruleTerms(numInputs);
        for (std::size_t j = numInputs; j > 0; --j)
        {
            std::size_t jj = j-1;
            const std::size_t termIdx = tmp % numInTerms_[jj];

            ruleTerms[jj] = termIdx;

            tmp = static_cast<std::size_t>(std::floor(tmp/static_cast<double>(numInTerms_[jj])));
        }
        for (std::size_t j = 0; j < numInputs; ++j)
        {
            const fl::InputVariable* p_iv = p_fis->getInputVariable(j);

            oss << p_iv->getName() << " " << fl::Rule::isKeyword() << " " << p_iv->getTerm(ruleTerms[j])->getName() << " ";

            if (j < (numInputs-1))
            {
                oss << fl::Rule::andKeyword() << " ";
            }
        }
#else // FL_DEBUG
        for (std::size_t j = 0; j < numInputs; ++j)
        {
            const fl::InputVariable* p_iv = p_fis->getInputVariable(j);
            const std::size_t termIdx = tmp % numInTerms_[j];

            oss << p_iv->getName() << " " << fl::Rule::isKeyword() << " " << p_iv->getTerm(termIdx)->getName() << " ";

            if (j < (numInputs-1))
            {
                oss << fl::Rule::andKeyword() << " ";
            }

            tmp = static_cast<std::size_t>(std::floor(tmp/static_cast<double>(numInTerms_[j])));
        }
#endif // FL_DEBUG

        oss << fl::Rule::thenKeyword();

        for (std::size_t j = 0; j < numOutputs; ++j)
        {
            const fl::OutputVariable* p_ov = p_fis->getOutputVariable(j);
            oss << " " << p_ov->getName() << " " << fl::Rule::isKeyword() << " " << p_ov->getTerm(r)->getName();
            if (j < (numOutputs-1))
            {
                oss << " " << fl::Rule::andKeyword() << " ";
            }
        }
        p_rules->addRule(fl::Rule::parse(oss.str(), p_fis.get()));
    }
    p_fis->addRuleBlock(p_rules);

    return p_fis;
}

template <typename EngineT>
const std::size_t GridPartitionFisBuilder<EngineT>::DefaultNumberOfInputTerms = 2;

template <typename EngineT>
const std::string GridPartitionFisBuilder<EngineT>::DefaultInputTerm = fl::Bell().className();

template <typename EngineT>
const std::size_t GridPartitionFisBuilder<EngineT>::DefaultNumberOfOutputTerms = 1;

template <typename EngineT>
const std::string GridPartitionFisBuilder<EngineT>::DefaultOutputTerm = fl::Linear().className();

} // Namespace fl

#endif // FL_FIS_BUILDER_GRID_PARTITION_H
/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */
