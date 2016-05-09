/**
 * \file anfis/training/training_algorithm.cpp
 *
 * \brief Definitions for the base ANFIS training algorithm class
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

#include <cstddef>
#include <fl/detail/math.h>
#include <fl/anfis/engine.h>
#include <fl/anfis/training/training_algorithm.h>
#include <fl/dataset.h>
#include <fl/detail/traits.h>


namespace fl { namespace anfis {

TrainingAlgorithm::TrainingAlgorithm(Engine* p_anfis)
: p_anfis_(p_anfis)
{
}

TrainingAlgorithm::~TrainingAlgorithm()
{
    // empty
}

void TrainingAlgorithm::setEngine(Engine* p_anfis)
{
    p_anfis_ = p_anfis;
}

Engine* TrainingAlgorithm::getEngine() const
{
    return p_anfis_;
}

fl::scalar TrainingAlgorithm::train(const fl::DataSet<fl::scalar>& data,
                                    std::size_t maxEpochs,
                                    fl::scalar errorGoal)
{
    this->reset();

    fl::scalar rmse = 0;
    for (std::size_t epoch = 0; epoch < maxEpochs; ++epoch)
    {
        FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch);

        rmse = this->trainSingleEpoch(data);

        FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch << " -> RMSE: " << rmse);

        if (fl::detail::FloatTraits<fl::scalar>::EssentiallyLessEqual(rmse, errorGoal))
        {
            break;
        }
    }
    return rmse;
}

fl::scalar TrainingAlgorithm::train(const fl::DataSet<fl::scalar>& trainData,
                                    const fl::DataSet<fl::scalar>& checkData,
                                    std::size_t maxEpochs,
                                    fl::scalar errorGoal)
{
    this->reset();

    fl::scalar minCheckRmse = std::numeric_limits<fl::scalar>::infinity();
    fl::anfis::Engine bestCheckAnfis;
    fl::scalar trainRmse = 0;
    fl::scalar checkRmse = 0;
    for (std::size_t epoch = 0; epoch < maxEpochs; ++epoch)
    {
        FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch);

        trainRmse = this->trainSingleEpoch(trainData);

        if (checkData.size() > 0)
        {

            for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = checkData.entryBegin(),
                                                                      entryEndIt = checkData.entryEnd();
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
                const std::vector<fl::scalar> anfisOut = this->getEngine()->eval(entry.inputBegin(), entry.inputEnd());

                for (std::size_t i = 0; i < nout; ++i)
                {
                    checkRmse += fl::detail::Sqr(targetOut[i]-anfisOut[i]);
                }
            }
            checkRmse = std::sqrt(checkRmse/checkData.size());

            if (checkRmse < minCheckRmse)
            {
                minCheckRmse = checkRmse;
                bestCheckAnfis = *(this->getEngine());
            }
        }

        FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch << " -> Train RMSE: " << trainRmse << ", Check RMSE: " << checkRmse << ", Best Check RMSE: " << minCheckRmse);

        if (fl::detail::FloatTraits<fl::scalar>::EssentiallyLessEqual(trainRmse, errorGoal))
        {
            break;
        }
    }

    if (checkData.size() > 0)
    {
        // Copy back the ANFIS engine with the best parameters wrt validation set

        *p_anfis_ = bestCheckAnfis;

        return checkRmse;
    }

    return trainRmse;
}

fl::scalar TrainingAlgorithm::trainSingleEpoch(const fl::DataSet<fl::scalar>& data)
{
    p_anfis_->setIsLearning(true);

    fl::scalar rmse = 0;

    rmse = this->doTrainSingleEpoch(data);

    p_anfis_->setIsLearning(false);

    return rmse;
}

void TrainingAlgorithm::reset()
{
    this->doReset();
}

}} // Namespace fl::anfis

/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */
