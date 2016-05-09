/**
 * \file fl/fis_builder/subtractive_clustering.h
 *
 * \brief Input partitioning method based on subtractive clustering (Chiu, 1994)
 *  for the fuzzy identification
 *
 * References
 * -# S.L. Chiu, "Fuzzy Model Identification Based on Cluster Estimation," Journal of Intelligent and Fuzzy Systems, 2(3):267-278, 1994
 * .
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
 *
 */

#ifndef FL_FIS_SUBTRACTIVE_CLUSTERINGARTITION_H
#define FL_FIS_SUBTRACTIVE_CLUSTERINGARTITION_H

#include <cstddef>
#include <fl/activation/General.h>
#include <fl/cluster/subtractive.h>
#include <fl/dataset.h>
#include <fl/defuzzifier/WeightedAverage.h>
#include <fl/detail/math.h>
#include <fl/fuzzylite.h>
#include <fl/macro.h>
#include <fl/norm/s/Maximum.h>
#include <fl/norm/t/AlgebraicProduct.h>
#include <fl/norm/s/AlgebraicSum.h>
#include <fl/rule/Rule.h>
#include <fl/rule/RuleBlock.h>
#include <fl/term/Accumulated.h> //FIXME: needed even if not explicitly used because of fwd decl in fl::OutputVariable
#include <fl/term/Gaussian.h>
#include <fl/term/Linear.h>
#include <fl/variable/InputVariable.h>
#include <fl/variable/OutputVariable.h>
#include <sstream>
#include <vector>


namespace fl {

/**
 * FIS builder based on the <em>subtractive clustering</em> method discussed in
 * (Chiu, 1994).
 *
 * With the <em>subtractive clustering</em> method, the multidimensional
 * input-output data space is divided into partitions where each cluster center
 * is used as the basis of a rule that describe a fuzzy system.
 *
 * References:
 * -# S.L. Chiu, "Fuzzy Model Identification Based on Cluster Estimation," Journal of Intelligent and Fuzzy Systems, 2(3):267-278, 1994
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename EngineT>
class SubtractiveClusteringFisBuilder
{
public:
    SubtractiveClusteringFisBuilder();

    SubtractiveClusteringFisBuilder(const fl::cluster::SubtractiveClustering& subclust);

    FL_unique_ptr<EngineT> build(const fl::DataSet<fl::scalar>& data);

    template <typename MatrixT>
    FL_unique_ptr<EngineT> build(const MatrixT& data, std::size_t numInputs, std::size_t numOutputs);


private:
    fl::cluster::SubtractiveClustering subclust_;
}; // SubtractiveClusteringFisBuilder


/////////////////////////
// Template definitiions
/////////////////////////


template <typename EngineT>
SubtractiveClusteringFisBuilder<EngineT>::SubtractiveClusteringFisBuilder()
{
}

template <typename EngineT>
SubtractiveClusteringFisBuilder<EngineT>::SubtractiveClusteringFisBuilder(const fl::cluster::SubtractiveClustering& subclust)
: subclust_(subclust)
{
}

template <typename EngineT>
FL_unique_ptr<EngineT> SubtractiveClusteringFisBuilder<EngineT>::build(const fl::DataSet<fl::scalar>& data)
{
    return this->build(data.data(), data.numOfInputs(), data.numOfOutputs());
}

template <typename EngineT>
template <typename MatrixT>
FL_unique_ptr<EngineT> SubtractiveClusteringFisBuilder<EngineT>::build(const MatrixT& data, std::size_t numInputs, std::size_t numOutputs)
{
    //const std::size_t numInOuts = numInputs+numOutputs;
    const std::size_t numData = data.size();

    subclust_.reset();
    subclust_.cluster(data);

    const std::vector< std::vector<fl::scalar> > centers = subclust_.centers();
    const std::vector<fl::scalar> sigmas = subclust_.rangeOfInfluence();
    const std::size_t numRules = centers.size();

    std::vector<fl::scalar> distFactors(numInputs);
    const fl::scalar invSqrt2 = 1.0/std::sqrt(2.0);
    for (std::size_t i = 0; i < numInputs; ++i)
    {
        distFactors[i] = invSqrt2 / sigmas[i];
    }


    // Computes values for eq. (4) and (5) in (Chiu,1994)

    std::vector<fl::scalar> sumMuValues(numData);
    std::vector<fl::scalar> invSumMuValues(numData);
    std::vector< std::vector<fl::scalar> > muMatrix(numData);
    const std::size_t muMatrixNumCols = numRules*(numInputs+1);
    for (std::size_t i = 0; i < numRules; ++i)
    {
        //std::vector< std::vector<fl::scalar> > sqDistMatrix(numData, numInputs);
        std::vector<fl::scalar> muValues(numData);
        for (std::size_t k = 0; k < numData; ++k)
        {
            fl::scalar sqDistSum = 0;
            for (std::size_t j = 0; j < numInputs; ++j)
            {
                const fl::scalar sqDist = fl::detail::Sqr((data[k][j]-centers[i][j])*distFactors[j]);

                //sqDistMatrix[k][j] = sqDist;
                sqDistSum += sqDist;
            }
            muValues[k] = std::exp(-sqDistSum);
            sumMuValues[k] += muValues[k];
        }

        const std::size_t offset = i*(numInputs+1);
        for (std::size_t k = 0; k < numData; ++k)
        {
            muMatrix[k].resize(muMatrixNumCols);

            for (std::size_t j = 0; j < numInputs; ++j)
            {
                muMatrix[k][j+offset] = data[k][j]*muValues[k];
            }
            //muMatrix[k][numInputs+offset+1] = muValues[k];
            muMatrix[k][numInputs+offset] = muValues[k];
        }
    }
    for (std::size_t i = 0; i < numData; ++i)
    {
        invSumMuValues[i] = 1.0/sumMuValues[i];

        for (std::size_t j = 0; j < muMatrixNumCols; ++j)
        {
            muMatrix[i][j] *= invSumMuValues[i];
        }
    }

    // Computes the Tagagi-Sugeno parameters by solving a linear least-squares estimation problem

    std::vector< std::vector<fl::scalar> > outParams; // The matrix of output parameters with dimension ((numRules*(numInputs+1) x numOutputs)
    {
        // Each column of outParams will contain the output equation parameters
        // for an output variable.  For example, if output variable y1 is given by
        // the equation y1 = k1*x1 + k2*x2 + k3*x3 + k0, then column 1 of
        // outParams contains [k1 k2 k3 k0] for rule #1, followed by [k1 k2 k3 k0]
        // for rule #2, etc.
 
        std::vector< std::vector<fl::scalar> > dataOut(numData);
        for (std::size_t i = 0; i < numData; ++i)
        {
            dataOut[i].resize(numOutputs);
            for (std::size_t j = 0; j < numOutputs; ++j)
            {
                dataOut[i][j] = data[i][j+numInputs];
            }
        }
        outParams = fl::detail::LsqSolveMulti<fl::scalar>(muMatrix, dataOut);
std::cerr << "muMatrix: "; fl::detail::MatrixOutput(std::cerr, muMatrix); std::cerr << std::endl;
std::cerr << "Xout: "; fl::detail::MatrixOutput(std::cerr, dataOut); std::cerr << std::endl;
std::cerr << "outEqns: "; fl::detail::MatrixOutput(std::cerr, outParams); std::cerr << std::endl;
    }

    const std::vector<fl::scalar> mins = subclust_.lowerBounds();
    const std::vector<fl::scalar> maxs = subclust_.upperBounds();

    FL_unique_ptr<EngineT> p_fis(new EngineT());

    // Generates input variables and terms
    std::vector<fl::InputVariable*> inputs(numInputs);
    for (std::size_t i = 0; i < numInputs; ++i)
    {
        std::ostringstream oss;

        fl::InputVariable* p_iv = new fl::InputVariable();

        oss << "in" << i;
        p_iv->setEnabled(true);
        p_iv->setName(oss.str());
        p_iv->setRange(mins[i], maxs[i]);

        for (std::size_t j = 0; j < numRules; ++j)
        {
            // Give a name to this term
            oss.str("");
            oss << p_iv->getName() << "mf" << j;

            p_iv->addTerm(new fl::Gaussian(oss.str(), centers[j][i], sigmas[i]));
        }

        p_fis->addInputVariable(p_iv);
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
        p_ov->fuzzyOutput()->setAccumulation(new fl::Maximum());
    //p_ov->fuzzyOutput()->setAccumulation(fl::null);
        p_ov->setDefuzzifier(new fl::WeightedAverage());
        p_ov->setDefaultValue(fl::nan);
        p_ov->setPreviousValue(false);

        for (std::size_t j = 0; j < numRules; ++j)
        {
            // Give a name to this term
            oss.str("");
            oss << p_ov->getName() << "mf" << j;

            const std::size_t numParams = numInputs+1;
            std::vector<fl::scalar> params(numParams);
            for (std::size_t k = 0; k < numParams; ++k)
            {
                params[k] = outParams[j*numParams+k][i];
            }

            p_ov->addTerm(new fl::Linear(oss.str(), params, p_fis.get()));
        }

        p_fis->addOutputVariable(p_ov);
    }

    // Generate rules
    fl::RuleBlock* p_rules = new fl::RuleBlock();
    p_rules->setEnabled(true);
    p_rules->setConjunction(new fl::AlgebraicProduct());
    p_rules->setDisjunction(new fl::AlgebraicSum());
    ////p_rules->setActivation(new fl::AlgebraicProduct());
    //p_rules->setActivation(fl::null);
    p_rules->setActivation(new fl::General());
    p_rules->setImplication(new fl::AlgebraicProduct());
    for (std::size_t r = 0; r < numRules; ++r)
    {
        std::ostringstream oss;

        oss << fl::Rule::ifKeyword() << " ";

        for (std::size_t j = 0; j < numInputs; ++j)
        {
            const fl::InputVariable* p_iv = p_fis->getInputVariable(j);

            oss << p_iv->getName() << " " << fl::Rule::isKeyword() << " " << p_iv->getTerm(r)->getName() << " ";

            if (j < (numInputs-1))
            {
                oss << fl::Rule::andKeyword() << " ";
            }
        }

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

} // Namespace fl

#endif // FL_FIS_BUILDER_SUBTRACTIVE_CLUSTERING_H
/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */
