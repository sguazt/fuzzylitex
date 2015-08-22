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
