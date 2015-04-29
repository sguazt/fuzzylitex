/**
 * \file fl/anfis/training/gradient_descent.h
 *
 * \brief Gradient descent backpropagation learning algorithm with momentum
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

#ifndef FL_ANFIS_TRAINING_GRADIENT_DESCENT_H
#define FL_ANFIS_TRAINING_GRADIENT_DESCENT_H


#include <cstddef>
#include <deque>
#include <fl/anfis/engine.h>
#include <fl/anfis/training/training_algorithm.h>
#include <fl/dataset.h>
#include <fl/detail/rls.h>
#include <fl/fuzzylite.h>
#include <map>
#include <vector>


namespace fl { namespace anfis {

/**
 * Gradient descent backpropagation learning algorithm with momentum
 *
 * In the backpropagation learning algorithm, there are two phases in its
 * learning cycle:
 * 1. the feedforward phase, where the input pattern is propagated through the
 *    network, and
 * 2. the backward phase, where the error between the desired and obtained
 *    output is back-propagated through the network in order to change the
 *    weights and then to decrease the error.
 *
 * This class implements both the batch (offline) and stochastic (online)
 * gradient descent backpropagation algorithm.
 *
 * References:
 * -# T.M. Mitchell, "Machine Learning," McGraw-Hill, 1997.
 * -# R. Rojas, "Neural Networks: A Sistematic Introduction," Springer, 1996.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API GradientDescentBackpropagationAlgorithm: public TrainingAlgorithm
{
private:
	typedef TrainingAlgorithm BaseType;


public:
	/**
	 * Constructor
	 *
	 * \param p_anfis Pointer to the ANFIS model to be trained
	 * \param ss The initial step size used in the parameter update formula
	 * \param ssDecrRate The step size decrease rate
	 * \param ssIncrRate The step size increase rate
	 * \param ff The forgetting factor used in the recursive least square
	 *  algorithm
	 */
    explicit GradientDescentBackpropagationAlgorithm(Engine* p_anfis = fl::null,
													 fl::scalar ss = 0.01,
													 fl::scalar ssDecrRate = 0.9,
													 fl::scalar ssIncrRate = 1.1);

	/// Sets the initial step size
	void setInitialStepSize(fl::scalar value);

	/// Gets the initial step size
	fl::scalar getInitialStepSize() const;

	/// Sets the step size decrease rate
	void setStepSizeDecreaseRate(fl::scalar value);

	/// Gets the step size decrease rate
	fl::scalar getStepSizeDecreaseRate() const;

	/// Sets the step size increase rate
	void setStepSizeIncreaseRate(fl::scalar value);

	/// Gets the step size increase rate
	fl::scalar getStepSizeIncreaseRate() const;

	///// Sets the momentum value
	//void setMomentum(fl::scalar value);

	///// Gets the momentum value
	//fl::scalar getMomentum() const;

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
	fl::scalar trainSingleEpochOffline(const fl::DataSet<fl::scalar>& data);

	/// Trains ANFIS for a signle epoch in online mode
	fl::scalar trainSingleEpochOnline(const fl::DataSet<fl::scalar>& data);

	/// Updates parameters of input terms
	void updateInputParameters();

	/// Updates the step-size (and the learning rate as well)
	void updateStepSize();

	/// Resets state for single epoch training
	void resetSingleEpoch();

	/// Gets the number of parameters of each output term
	std::size_t numberOfOutputTermParameters() const;

    /// Trains the ANFIS model for a single epoch only using the given training set \a data
    fl::scalar doTrainSingleEpoch(const fl::DataSet<fl::scalar>& data);

	/// Resets the state of the learning algorithm
	void doReset();


private:
	fl::scalar stepSizeInit_; ///< The initial value of the step size
	fl::scalar stepSizeDecrRate_; ///< The rate at which the step size must be decreased
	fl::scalar stepSizeIncrRate_; ///< The rate at which the step size must be increased
	fl::scalar stepSize_; ///< Step size to use in the parameter update formula representing the length of each transition along the gradient direction in the parameter space
	std::size_t stepSizeErrWindowLen_; ///< Length of the RMSE window used to update the step size
	std::deque<fl::scalar> stepSizeErrWindow_; ///< Window of RMSEs used to update the step size
	std::size_t stepSizeIncrCounter_; ///< Counter used to check when to increase the step size
	std::size_t stepSizeDecrCounter_; ///< Counter used to check when to decrease the step size
	bool online_; ///< \c true in case of online learning; \c false if offline (batch) learning
	//fl::scalar momentum_; ///< The momentum constant for momentum learning
	std::map< Node*, std::vector<fl::scalar> > dEdPs_; ///< Error derivatives wrt node parameters
	std::map< Node*, std::vector<fl::scalar> > oldDeltaPs_; ///< Old values of parameters changes (only for momentum learning)
}; // GradientDescentBackpropagationAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_GRADIENT_DESCENT_H
