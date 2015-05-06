/**
 * \file fl/anfis/training/training_algorithm.h
 *
 * \brief Base class for ANFIS training algorithms
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

#ifndef FL_ANFIS_TRAINING_TRAINING_ALGORITHM_H
#define FL_ANFIS_TRAINING_TRAINING_ALGORITHM_H


#include <cstddef>
#include <fl/anfis/engine.h>
#include <fl/dataset.h>
#include <fl/fuzzylite.h>


namespace fl { namespace anfis {

/**
 * Base class for ANFIS learning algorithms
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API TrainingAlgorithm
{
public:
    /**
     * Constructor
     *
     * \param p_anfis Pointer to the ANFIS model to be trained
     *  algorithm
     */
    explicit TrainingAlgorithm(Engine* p_anfis = fl::null);

    /// Sets the ANFIS model to be trained
    void setEngine(Engine* p_anfis);

    /// Gets the ANFIS model to be trained
    Engine* getEngine() const;

    /**
     * Trains the ANFIS model
     *
     * \param data The training set
     * \param maxEpochs The maximum number of epochs
     * \param errorGoal The error to achieve
     *
     * The error measure is the Root Mean Squared Error (RMSE).
     *
     * \return The achieved error
     */
    fl::scalar train(const fl::DataSet<fl::scalar>& data,
                     std::size_t maxEpochs = 10,
                     fl::scalar errorGoal = 0);

    /// Trains the ANFIS model for a single epoch only using the given training set \a data
    fl::scalar trainSingleEpoch(const fl::DataSet<fl::scalar>& data);

    /// Resets the state of the learning algorithm
    void reset();

private:
    /// Trains the ANFIS model for a single epoch only using the given training set \a data
    virtual fl::scalar doTrainSingleEpoch(const fl::DataSet<fl::scalar>& data) = 0;

    /// Resets state for single epoch training
    virtual void doReset() = 0;


private:
    Engine* p_anfis_; ///< The ANFIS model
}; // TraningAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_TRAINING_ALGORITHM_H
