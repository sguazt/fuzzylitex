/**
 * \file fl/anfis/training/least_squares.h
 *
 * \brief Least-squares training algorithm.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2016 Marco Guazzone (marco.guazzone@gmail.com)
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

#ifndef FL_ANFIS_TRAINING_LEAST_SQUARES_H
#define FL_ANFIS_TRAINING_LEAST_SQUARES_H


#include <cstddef>
#include <deque>
#include <fl/anfis/engine.h>
#include <fl/anfis/training/training_algorithm.h>
#include <fl/dataset.h>
//#include <fl/detail/kalman.h>
#include <fl/detail/rls.h>
#include <fl/fuzzylite.h>
#include <map>
#include <vector>


namespace fl { namespace anfis {

/**
 * Least-squares learning algorithm
 *
 * The least-squares learning algorithm uses least-squares estimation to
 * identify the parameters of the input and output membership functions of a
 * Sugeno-type fuzzy inference system.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API LeastSquaresLearningAlgorithm: public TrainingAlgorithm
{
private:
    typedef TrainingAlgorithm BaseType;


public:
    /**
     * Constructor
     *
     * \param p_anfis Pointer to the ANFIS model to be trained
     * \param ff The forgetting factor used in the recursive least square
     *  algorithm
     */
    explicit LeastSquaresLearningAlgorithm(Engine* p_anfis = fl::null,
                                  		   fl::scalar ff = 1);

    /// Sets the forgetting factor
    void setForgettingFactor(fl::scalar value);

    /// Gets the forgetting factor
    fl::scalar getForgettingFactor() const;

    /// Sets the online/offline mode for the learning algorithm
    void setIsOnline(bool value);

    /// Gets the online/offline mode of the learning algorithm
    bool isOnline() const;

private:
    /// Initializes the training algorithm
    void init();

    /// Checks the correctness of the parameters of the training algorithm
    void check() const;

    /// Trains ANFIS for a signle epoch in offline (batch) mode
    fl::scalar trainSingleEpochOffline(const fl::DataSet<fl::scalar>& trainData);

    /// Trains ANFIS for a signle epoch in online mode
    fl::scalar trainSingleEpochOnline(const fl::DataSet<fl::scalar>& trainData);

    /// Updates parameters of input terms
    void updateInputParameters();

    /// Resets state for single epoch training
    void resetSingleEpoch();

    /// Gets the number of parameters of each output term
    std::size_t numberOfOutputTermParameters() const;

    /// Trains the ANFIS model for a single epoch only using the given training set \a trainData
    fl::scalar doTrainSingleEpoch(const fl::DataSet<fl::scalar>& trainData);

    /// Resets the state of the learning algorithm
    void doReset();


private:
    bool online_; ///< \c true in case of online learning; \c false if offline (batch) learning
    fl::detail::RecursiveLeastSquaresEstimator<fl::scalar> rls_; ///< The recursive least-squares estimator
}; // LeastSquaresLearningAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_LEAST_SQUARES_H
