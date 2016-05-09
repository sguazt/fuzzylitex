/**
 * \file anfis/training/least_squares.cpp
 *
 * \brief Definitions for the (Jang et al.,1993) ANFIS hybrid learning algorithm
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

#include <cmath>
#include <cstddef>
#include <fl/anfis/engine.h>
#include <fl/anfis/training/least_squares.h>
#include <fl/dataset.h>
#include <fl/detail/math.h>
//#include <fl/detail/kalman.h>
#include <fl/detail/rls.h>
#include <fl/detail/terms.h>
#include <fl/detail/traits.h>
#include <fl/fuzzylite.h>
#include <fl/Operation.h>
#include <fl/term/Linear.h>
#include <fl/term/Term.h>
#include <fl/variable/OutputVariable.h>
#include <map>
#include <vector>


namespace fl { namespace anfis {

LeastSquaresLearningAlgorithm::LeastSquaresLearningAlgorithm(Engine* p_anfis,
                                                             fl::scalar ff)
: BaseType(p_anfis),
  online_(false),
  rls_(0,0,0,ff)/*,
  minCheckRmse_(std::numeric_limits<fl::scalar>::infinity())*/
{
    this->init();
}

void LeastSquaresLearningAlgorithm::setForgettingFactor(fl::scalar value)
{
    rls_.setForgettingFactor(value);
}

fl::scalar LeastSquaresLearningAlgorithm::getForgettingFactor() const
{
    return rls_.getForgettingFactor();
}

void LeastSquaresLearningAlgorithm::setIsOnline(bool value)
{
    online_ = value;
}

bool LeastSquaresLearningAlgorithm::isOnline() const
{
    return online_;
}

fl::scalar LeastSquaresLearningAlgorithm::doTrainSingleEpoch(const fl::DataSet<fl::scalar>& trainData)
{
    this->check();

    fl::scalar rmse = 0;

    if (online_)
    {
        rmse = this->trainSingleEpochOnline(trainData);
    }
    else
    {
        rmse = this->trainSingleEpochOffline(trainData);
    }

    return rmse;
}

void LeastSquaresLearningAlgorithm::doReset()
{
    this->init();
}

fl::scalar LeastSquaresLearningAlgorithm::trainSingleEpochOffline(const fl::DataSet<fl::scalar>& trainData)
{
    this->resetSingleEpoch();

    const std::size_t numOutTermParams = this->numberOfOutputTermParameters();

    fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch
    std::size_t numTrainings = 0;

    // Forwards inputs from input layer to antecedent layer, and estimate parameters with RLS
    //std::vector< std::vector<fl::scalar> > antecedentValues;
    for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = trainData.entryBegin(),
                                                              entryEndIt = trainData.entryEnd();
         entryIt != entryEndIt;
         ++entryIt)
    {
        const fl::DataSetEntry<fl::scalar>& entry = *entryIt;

        const std::size_t nout = entry.numOfOutputs();

        if (nout != this->getEngine()->numberOfOutputVariables())
        {
            FL_THROW2(std::invalid_argument, "Incorrect output dimension");
        }

        const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

        // Compute current rule firing strengths
        const std::vector<fl::scalar> ruleFiringStrengths = this->getEngine()->evalTo(entry.inputBegin(), entry.inputEnd(), fl::anfis::Engine::AntecedentLayer);

        // Compute input to RLS algorithm
        std::vector<fl::scalar> rlsInputs(rls_.getInputDimension());
        {
            // Compute normalization factor
            const fl::scalar totRuleFiringStrength = fl::detail::Sum<fl::scalar>(ruleFiringStrengths.begin(), ruleFiringStrengths.end());
//{//[XXX]
//std::cerr << "PHASE #0 - Rule firing strengths: "; fl::detail::VectorOutput(std::cerr, ruleFiringStrengths); std::cerr << std::endl;
//std::cerr << "PHASE #0 - Total Rule firing strength: " << totRuleFiringStrength << std::endl;
//std::cerr << "PHASE #0 - Normalized Rule firing strength: ";
//for (std::size_t i = 0; i < ruleFiringStrengths.size(); ++i)
//{
//  std::cerr << ruleFiringStrengths[i]/totRuleFiringStrength << " ";
//}
//std::cerr << std::endl;
//}//[XXX]

            if (totRuleFiringStrength <= 0)
            {
                // No rule is active -> skip this training data
                continue;
            }

            for (std::size_t i = 0,
                             ni = this->getEngine()->numberOfRuleBlocks();
                 i < ni;
                 ++i)
            {
                fl::RuleBlock* p_rb = this->getEngine()->getRuleBlock(i);

                // check: null
                FL_DEBUG_ASSERT( p_rb );

                std::size_t k = 0;
                for (std::size_t r = 0,
                                 nr = p_rb->numberOfRules();
                     r < nr;
                     ++r)
                {
                    for (std::size_t p = 1; p < numOutTermParams; ++p)
                    {
                        rlsInputs[k] = ruleFiringStrengths[r]*entry.getInput(p-1)/totRuleFiringStrength;
                        ++k;
                    }
                    rlsInputs[k] = ruleFiringStrengths[r]/totRuleFiringStrength;
                    ++k;
                }
            }
        }
//std::cerr << "PHASE #0 - Num inputs: " << rls_.getInputDimension() << " - Num Outputs: " << rls_.getOutputDimension() << " - Order: " << rls_.getModelOrder() << std::endl;//XXX
//std::cerr << "PHASE #0 - RLS Input: "; fl::detail::VectorOutput(std::cerr, rlsInputs); std::cerr << std::endl;///XXX
        // Estimate parameters
        std::vector<fl::scalar> actualOut;
        actualOut = rls_.estimate(rlsInputs.begin(), rlsInputs.end(), targetOut.begin(), targetOut.end());
//std::cerr << "PHASE #0 - Target: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - Actual: ";fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << std::endl;///XXX

        ++numTrainings;
    }

    // Put estimated RLS parameters in the ANFIS model
    if (numTrainings > 0)
    {
        const std::vector< std::vector<fl::scalar> > rlsParamMatrix = rls_.getEstimatedParameters();
//std::cerr << "PHASE #0 - Estimated RLS params: "; fl::detail::MatrixOutput(std::cerr, rlsParamMatrix); std::cerr << std::endl;//XXX

        for (std::size_t v = 0,
                         nv = this->getEngine()->numberOfOutputVariables();
             v < nv;
             ++v)
        {
            fl::OutputVariable* p_var = this->getEngine()->getOutputVariable(v);

            FL_DEBUG_ASSERT( p_var );

            std::size_t k = 0;
            for (std::size_t t = 0,
                             nt = p_var->numberOfTerms();
                 t < nt;
                 ++t)
            {
                fl::Term* p_term = p_var->getTerm(t);

                FL_DEBUG_ASSERT( p_term );

                const std::size_t numParams = detail::GetTermParameters(p_term).size();
                std::vector<fl::scalar> params(numParams);
                for (std::size_t p = 0; p < numParams; ++p)
                {
                    params[p] = rlsParamMatrix[k][v];
                    ++k;
                }
                detail::SetTermParameters(p_term, params.begin(), params.end());
//std::cerr << "PHASE #0 - Estimated RLS params - Output #" << v << " - Term #" << t << " - Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;//XXX
                //++r;
            }
        }

        //rlsPhi_ = rls_.getRegressor();
    }

    for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = trainData.entryBegin(),
                                                              entryEndIt = trainData.entryEnd();
         entryIt != entryEndIt;
         ++entryIt)
    {
        const fl::DataSetEntry<fl::scalar>& entry = *entryIt;
//std::cerr << "PHASE #1 - Traning data #: " << std::distance(trainData.entryBegin(), entryIt) << std::endl;//XXX

        const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

        // Compute ANFIS output
        const std::vector<fl::scalar> actualOut = this->getEngine()->eval(entry.inputBegin(), entry.inputEnd());

        // Update bias in case of zero rule firing strength
        if (this->getEngine()->hasBias())
        {
            bool skip = false;

            for (std::size_t i = 0,
                             ni = targetOut.size();
                 i < ni;
                 ++i)
            {
                if (fl::Operation::isNaN(actualOut[i]))
                {
                    OutputNode* p_outNode = this->getEngine()->getOutputLayer().at(i);

                    FL_DEBUG_ASSERT( p_outNode );

                    fl::scalar bias = p_outNode->getBias();
                    bias += (targetOut[i]-bias);
                    p_outNode->setBias(bias);
                    skip = true;
                }
            }
            //this->getEngine()->setBias(bias_);

            if (skip)
            {
                // Skip this data point
//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, this->getEngine()->getBias()); std::cerr << std::endl; //XXX
                continue;
            }
        }

//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, this->getEngine()->getBias()); std::cerr << std::endl; //XXX

        // Update error
        fl::scalar squaredErr = 0;
        for (std::size_t i = 0,
                         ni = targetOut.size();
             i < ni;
             ++i)
        {
            const fl::scalar out = fl::Operation::isNaN(actualOut[i]) ? 0.0 : actualOut[i];

            squaredErr += fl::detail::Sqr(targetOut[i]-out);
        }
        rmse += squaredErr;
    }

    //rmse = std::sqrt(rmse/trainData.size());
    rmse = std::sqrt(rmse/numTrainings);

    return rmse;
}

fl::scalar LeastSquaresLearningAlgorithm::trainSingleEpochOnline(const fl::DataSet<fl::scalar>& trainData)
{
    //rls_.reset();
    this->resetSingleEpoch();

    const std::size_t numOutTermParams = this->numberOfOutputTermParameters();

    fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch
    std::size_t numTrainings = 0;

    // Forwards inputs from input layer to antecedent layer, and estimate parameters with RLS
    //std::vector< std::vector<fl::scalar> > antecedentValues;
    for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = trainData.entryBegin(),
                                                              entryEndIt = trainData.entryEnd();
         entryIt != entryEndIt;
         ++entryIt)
    {
        const fl::DataSetEntry<fl::scalar>& entry = *entryIt;

        const std::size_t nout = entry.numOfOutputs();

        if (nout != this->getEngine()->numberOfOutputVariables())
        {
            FL_THROW2(std::invalid_argument, "Incorrect output dimension");
        }

        const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

        // Compute current rule firing strengths
        const std::vector<fl::scalar> ruleFiringStrengths = this->getEngine()->evalTo(entry.inputBegin(), entry.inputEnd(), fl::anfis::Engine::AntecedentLayer);

        // Compute input to RLS algorithm
        std::vector<fl::scalar> rlsInputs(rls_.getInputDimension());
        {
            // Compute normalization factor
            const fl::scalar totRuleFiringStrength = fl::detail::Sum<fl::scalar>(ruleFiringStrengths.begin(), ruleFiringStrengths.end());

            if (totRuleFiringStrength <= 0)
            {
                // No rule is active -> skip this training data
                continue;
            }

            for (std::size_t i = 0,
                             ni = this->getEngine()->numberOfRuleBlocks();
                 i < ni;
                 ++i)
            {
                fl::RuleBlock* p_rb = this->getEngine()->getRuleBlock(i);

                // check: null
                FL_DEBUG_ASSERT( p_rb );

                std::size_t k = 0;
                for (std::size_t r = 0,
                                 nr = p_rb->numberOfRules();
                     r < nr;
                     ++r)
                {
                    for (std::size_t p = 1; p < numOutTermParams; ++p)
                    {
                        rlsInputs[k] = ruleFiringStrengths[r]*entry.getInput(p-1)/totRuleFiringStrength;
                        ++k;
                    }
                    rlsInputs[k] = ruleFiringStrengths[r]/totRuleFiringStrength;
                    ++k;
                }
            }
        }

        // Estimate parameters
        std::vector<fl::scalar> actualOut;
        actualOut = rls_.estimate(rlsInputs.begin(), rlsInputs.end(), targetOut.begin(), targetOut.end());

        ++numTrainings;

        // Put estimated RLS parameters in the ANFIS model
        if (numTrainings > 0)
        {
            const std::vector< std::vector<fl::scalar> > rlsParamMatrix = rls_.getEstimatedParameters();
//std::cerr << "PHASE #0 - Estimated RLS params: "; fl::detail::MatrixOutput(std::cerr, rlsParamMatrix); std::cerr << std::endl;//XXX

            //std::size_t k = 0;
            ////std::size_t r = 0;
            for (std::size_t v = 0,
                             nv = this->getEngine()->numberOfOutputVariables();
                 v < nv;
                 ++v)
            {
                fl::OutputVariable* p_var = this->getEngine()->getOutputVariable(v);

                FL_DEBUG_ASSERT( p_var );

                std::size_t k = 0;
                for (std::size_t t = 0,
                                 nt = p_var->numberOfTerms();
                     t < nt;
                     ++t)
                {
                    fl::Term* p_term = p_var->getTerm(t);

                    FL_DEBUG_ASSERT( p_term );

                    const std::size_t numParams = detail::GetTermParameters(p_term).size();
                    std::vector<fl::scalar> params(numParams);
                    for (std::size_t p = 0; p < numParams; ++p)
                    {
                        params[p] = rlsParamMatrix[k][v];
                        ++k;
                    }
                    detail::SetTermParameters(p_term, params.begin(), params.end());
//std::cerr << "PHASE #0 - Estimated RLS params - Output #" << v << " - Term #" << t << " - Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;//XXX
                    //++r;
                }
            }

            //rlsPhi_ = rls_.getRegressor();
        }

        // Compute ANFIS output with the new estimated consequent parameters
        actualOut = this->getEngine()->eval(entry.inputBegin(), entry.inputEnd());
//std::cerr << "PHASE #0 - Target: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - Actual: ";fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << std::endl;///XXX

        // Update bias in case of zero rule firing strength
        if (this->getEngine()->hasBias())
        {
            bool skip = false;

            for (std::size_t i = 0,
                             ni = targetOut.size();
                 i < ni;
                 ++i)
            {
                if (fl::Operation::isNaN(actualOut[i]))
                {
                    OutputNode* p_outNode = this->getEngine()->getOutputLayer().at(i);

                    FL_DEBUG_ASSERT( p_outNode );

                    fl::scalar bias = p_outNode->getBias();
                    bias += (targetOut[i]-bias);
                    p_outNode->setBias(bias);
                    skip = true;
                }
            }
            //this->getEngine()->setBias(bias_);

            if (skip)
            {
                // Skip this data point
//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, this->getEngine()->getBias()); std::cerr << std::endl; //XXX
                continue;
            }
        }

//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, this->getEngine()->getBias()); std::cerr << std::endl; //XXX

        // Update error
        fl::scalar squaredErr = 0;
        for (std::size_t i = 0,
                         ni = targetOut.size();
             i < ni;
             ++i)
        {
            const fl::scalar out = fl::Operation::isNaN(actualOut[i]) ? 0.0 : actualOut[i];

            squaredErr += fl::detail::Sqr(targetOut[i]-out);
        }
        rmse += squaredErr;
    }

    //rmse = std::sqrt(rmse/trainData.size());
    rmse = std::sqrt(rmse/numTrainings);

    return rmse;
}

void LeastSquaresLearningAlgorithm::resetSingleEpoch()
{
    rls_.reset();
}

void LeastSquaresLearningAlgorithm::init()
{
    std::size_t numParams = 0;
    std::size_t numOutVars = 0;
    if (this->getEngine())
    {
        const std::size_t numTermParams = this->numberOfOutputTermParameters();

        for (std::size_t i = 0,
                         ni = this->getEngine()->numberOfRuleBlocks();
             i < ni;
             ++i)
        {
            fl::RuleBlock* p_rb = this->getEngine()->getRuleBlock(i);

            // check: null
            FL_DEBUG_ASSERT( p_rb );

            if (p_rb->isEnabled())
            {
                numParams += p_rb->numberOfRules()*numTermParams;
            }
        }

        numOutVars = this->getEngine()->numberOfOutputVariables();
    }

    rls_.setModelOrder(0);
    rls_.setInputDimension(numParams);
    rls_.setOutputDimension(numOutVars);
    rls_.reset();
    //rlsPhi_.clear();
}

void LeastSquaresLearningAlgorithm::check() const
{
    if (this->getEngine() == fl::null)
    {
        FL_THROW2(std::logic_error, "Invalid ANFIS engine");
    }
    if (this->getEngine()->type() != fl::Engine::TakagiSugeno)
    {
        FL_THROW2(std::logic_error, "This learning algorithm currently works only for Takagi-Sugeno ANFIS");
    }

    // Output terms must be homogeneous in shape
    {
        std::string termClass;
        for (std::size_t i = 0,
                         ni = this->getEngine()->numberOfOutputVariables();
             i < ni;
             ++i)
        {
            const fl::OutputVariable* p_var = this->getEngine()->getOutputVariable(i);

            // check: null
            FL_DEBUG_ASSERT( p_var );

            for (std::size_t j = 0,
                             nj = p_var->numberOfTerms();
                 j < nj;
                 ++j)
            {
                const fl::Term* p_term = p_var->getTerm(j);

                // check: null
                FL_DEBUG_ASSERT( p_term );

                if (termClass.empty())
                {
                    termClass = p_term->className();
                }
                else if (termClass != p_term->className())
                {
                    FL_THROW2(std::logic_error, "Output terms must be homogeneous (i.e., output membership functions must have the same shape)");
                }
            }
        }
    }
}

std::size_t LeastSquaresLearningAlgorithm::numberOfOutputTermParameters() const
{
    std::size_t numTermParams = 0;

    if (this->getEngine()->numberOfOutputVariables() > 0)
    {
        numTermParams += 1; // constant term

        const fl::OutputVariable* p_var = this->getEngine()->getOutputVariable(0);
        if (p_var->numberOfTerms() > 0)
        {
            const fl::Term* p_term = p_var->getTerm(0);

            if (p_term->className() == fl::Linear().className())
            {
                numTermParams += this->getEngine()->numberOfInputVariables();
            }
        }
    }

    return numTermParams;
}

}} // Namespace fl::anfis

/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */
