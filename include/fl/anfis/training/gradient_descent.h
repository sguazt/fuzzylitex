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
 * Base class for gradient descent backpropagation learning algorithms
 *
 * In the backpropagation learning algorithm, there are two phases in its
 * learning cycle:
 * 1. the feedforward phase, where the input pattern is propagated through the
 *    network, and
 * 2. the backward phase, where the error between the desired and obtained
 *    output is back-propagated through the network in order to change the
 *    weights and then to decrease the error.
 *
 * In order to minimize the error measure, the gradient descent backpropagation
 * algorithm updates the model parameter according to the so called
 * <em>generalized delta rule</em>, whereby, at the \f$n\f$-th iteration of the
 * algorithm, the parameters \f$\alpha\f$ are updated according to gradient of
 * the error measure (e.g., see [1,2]):
 * \f{align}
 *  \alpha(n) &= \alpha(n-1) + \Delta \alpha(n)
 * \f}
 * Various version of gradient descent backpropagation algorithms differ by the
 * way \f$\Delta \alpha(n)\f$ is computed.
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
	 */
    explicit GradientDescentBackpropagationAlgorithm(Engine* p_anfis = fl::null);

	/// Sets the online/offline mode for the learning algorithm
	void setIsOnline(bool value);

	/// Gets the online/offline mode of the learning algorithm
	bool isOnline() const;

protected:
	/// Resets the state of the learning algorithm
	virtual void doReset();

	/// Sets the current value of the error measure computed in the current epoch
	void setCurrentError(fl::scalar value);

	/// Gets the current value of the error measure computed in the current epoch
	fl::scalar getCurrentError() const;

	/// Gets the error derivatives wrt node parameters
	const std::map< Node*, std::vector<fl::scalar> >& getErrorDerivatives() const;

	/// Updates the bias of output nodes
	bool updateBias(const std::vector<fl::scalar>& targetOut, const std::vector<fl::scalar>& actualOut);

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

	/// Resets state for single epoch training
	void resetSingleEpoch();

    /// Trains the ANFIS model for a single epoch only using the given training set \a data
    fl::scalar doTrainSingleEpoch(const fl::DataSet<fl::scalar>& data);

	/// Updates parameters of input terms
	virtual void doUpdateInputParameters() = 0;

	/// Updates the bias of output nodes
	virtual bool doUpdateBias(const std::vector<fl::scalar>& targetOut, const std::vector<fl::scalar>& actualOut) = 0;

	/// Resets state for single epoch training
	virtual void doResetSingleEpoch() = 0;

	/// Checks the correctness of the parameters of the training algorithm
	virtual void doCheck() const = 0;


private:
	bool online_; ///< \c true in case of online learning; \c false if offline (batch) learning
	std::map< Node*, std::vector<fl::scalar> > dEdPs_; ///< Error derivatives wrt node parameters
	fl::scalar curError_; ///< The current value of the error measure
}; // GradientDescentBackpropagationAlgorithm

/**
 * Gradient descent backpropagation learning algorithm with adaptive learning
 * rate based on the step-size, proposed by (J.-S.R. Jang,1993)
 *
 * In the backpropagation learning algorithm, there are two phases in its
 * learning cycle:
 * 1. the feedforward phase, where the input pattern is propagated through the
 *    network, and
 * 2. the backward phase, where the error between the desired and obtained
 *    output is back-propagated through the network in order to change the
 *    weights and then to decrease the error.
 *
 * In order to minimize the error measure, the gradient descent backpropagation
 * algorithm updates the model parameter according to the so called
 * <em>generalized delta rule</em>, whereby, at the \f$n\f$-th iteration of the
 * algorithm, the parameters \f$\alpha\f$ are updated according to gradient of
 * the error measure (e.g., see [3,4]):
 * \f{align}
 *  \alpha(n) &= \alpha(n-1) + \Delta \alpha(n),\\
 *  \Delta \alpha(n) = - \eta(n) \frac{\partial E}{\partial \alpha}
 * \f}
 * where:
 * - \f$\eta(n)\f$ is a positive value called <em>learning rate</em> (whose
 *   purpose is to limit the degree to which weights are changed at each step),
 * - \f$E\f$ is the overall error measure that we want to minimize.
 * .
 *
 * This version of the gradient descent backpropagation algorithm uses an
 * adaptive learning rate strategy which is based on the <em>step-size</em>, as
 * discussed in [1,2], such that, at the \f$n\f$-th iteration of the algorithm,
 * the learning rate \f$\eta(n)\f$ is updated according to the following formula
 * (see [1,2]):
 * \f[
 *  \eta(n) = \frac{\kappa(n)}{\sqrt{\sum_\alpha \frac{\partial E}{\partial \alpha}}}
 * \f]
 * where \f$\kappa(n)\f$ is the <em>step-size</em> updated at the \f$n\f$-th
 * iteration of the algorithm.
 * The <em>step-size</em> is updated according to the following strategy:
 * - If the error undergoes four consecutive reductions, increase the step-size
 *   by multiplying it by a constant numeric value greater than one.
 * - If the error undergoes two consecutive combinations of one increase and one
 *   reduction, decrease the step-size by multiplying it by a constant numeric
 *   value less than one.
 * .
 *
 * This class implements both the batch (offline) and stochastic (online)
 * gradient descent backpropagation algorithm.
 *
 * References:
 * -# J.-S.R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference Systems," IEEE Transactions on Systems, Man, and Cybernetics, 23:3(665-685), 1993.
 * -# J.-S.R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
 * -# T.M. Mitchell, "Machine Learning," McGraw-Hill, 1997.
 * -# R. Rojas, "Neural Networks: A Sistematic Introduction," Springer, 1996.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API Jang1993GradientDescentBackpropagationAlgorithm: public GradientDescentBackpropagationAlgorithm
{
private:
	typedef GradientDescentBackpropagationAlgorithm BaseType;


public:
	/**
	 * Constructor
	 *
	 * \param p_anfis Pointer to the ANFIS model to be trained
	 * \param ss The initial step size used in the parameter update formula
	 * \param ssDecrRate The step size decrease rate
	 * \param ssIncrRate The step size increase rate
	 */
    explicit Jang1993GradientDescentBackpropagationAlgorithm(Engine* p_anfis = fl::null,
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

private:
	/// Initializes the training algorithm
	void init();

	/// Updates the step-size (and the learning rate as well)
	void updateStepSize();

	/// Checks the correctness of the parameters of the training algorithm
	void doCheck() const;

	/// Gets the number of parameters of each output term
	std::size_t numberOfOutputTermParameters() const;

	/// Updates parameters of input terms
	void doUpdateInputParameters();

	/// Updates the bias of output nodes
	bool doUpdateBias(const std::vector<fl::scalar>& targetOut, const std::vector<fl::scalar>& actualOut);

	/// Resets state for single epoch training
	void doResetSingleEpoch();

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
}; // Jang1993GradientDescentBackpropagationAlgorithm


/**
 * Gradient descent with momentum backpropagation learning algorithm
 *
 * In the backpropagation learning algorithm, there are two phases in its
 * learning cycle:
 * 1. the feedforward phase, where the input pattern is propagated through the
 *    network, and
 * 2. the backward phase, where the error between the desired and obtained
 *    output is back-propagated through the network in order to change the
 *    weights and then to decrease the error.
 *
 * In order to minimize the error measure, the gradient descent backpropagation
 * algorithm updates the model parameter according to the so called
 * <em>generalized delta rule</em>, whereby, at the \f$n\f$-th iteration of the
 * algorithm, the parameters \f$\alpha\f$ are updated according to gradient of
 * the error measure (e.g., see [1]):
 * \f{align}
 *  \alpha(n) &= \alpha(n-1) + \Delta \alpha(n),\\
 *  \Delta \alpha(n) = - (1-\mu)\eta \frac{\partial E}{\partial \alpha} + \mu \Delta \alpha(n-1)
 * \f}
 * where:
 * - \f$\eta\f$ is a positive constant called <em>learning rate</em> (whose
 *   purpose is to limit the degree to which weights are changed at each step),
 * - \f$\mu\f$ is a constant in [0,1) called <em>momentum</em> (whose purpose
 *   is to keep gradient descent search trajectory in the same direction from on
 *   iteration to the next), and
 * - \f$E\f$ is the overall error measure that we want to minimize.
 * .
 *
 * Theoretically, the use of the momemtum term should provide the search process
 * with a kind of inertia and could help to avoid excessive oscillations in
 * narrow valleys of the error function [3].
 *
 * The use of the momentum can sometimes have the effect of keeping the gradient
 * descent search trajectory through small local minima in the error surface, or
 * along flat regions in the surface where the gradient descent search
 * trajectory would stop if there were no momentum [2].
 * It also has the effect of gradually increasing the step size of the search in
 * regions where the gradient is unchanging, thereby speeding convergence [2].
 *
 * This class implements both the batch (offline) and stochastic (online)
 * gradient descent backpropagation algorithm.
 *
 * References:
 * -# M.T. Hagan et al., "Neural Network Design," PWS Publishing, 1996.
 * -# T.M. Mitchell, "Machine Learning," McGraw-Hill, 1997.
 * -# R. Rojas, "Neural Networks: A Sistematic Introduction," Springer, 1996.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API GradientDescentWithMomentumBackpropagationAlgorithm: public GradientDescentBackpropagationAlgorithm
{
private:
	typedef GradientDescentBackpropagationAlgorithm BaseType;


public:
	/**
	 * Constructor
	 *
	 * \param p_anfis Pointer to the ANFIS model to be trained
	 * \param learningRate The learning rate parameter
	 * \param momentum The momentum parameter (set to 0, to disable momentum learning)
	 */
    explicit GradientDescentWithMomentumBackpropagationAlgorithm(Engine* p_anfis = fl::null,
																 fl::scalar learningRate = 0.01,
																 fl::scalar momentum = 0.9);

	/// Sets the learning rate
	void setLearningRate(fl::scalar value);

	/// Gets the learning rate
	fl::scalar getLearningRate() const;

	/// Sets the momentum
	void setMomentum(fl::scalar value);

	/// Gets the momentum
	fl::scalar getMomentum() const;

private:
	/// Initializes the training algorithm
	void init();

	/// Checks the correctness of the parameters of the training algorithm
	void doCheck() const;

	/// Gets the number of parameters of each output term
	std::size_t numberOfOutputTermParameters() const;

	/// Updates parameters of input terms
	void doUpdateInputParameters();

	/// Updates the bias of output nodes
	bool doUpdateBias(const std::vector<fl::scalar>& targetOut, const std::vector<fl::scalar>& actualOut);

	/// Resets state for single epoch training
	void doResetSingleEpoch();

	/// Resets the state of the learning algorithm
	void doReset();


private:
	fl::scalar learnRate_; ///< The learning rate parameter
	fl::scalar momentum_; ///< The momentum parameter
	std::map< Node*, std::vector<fl::scalar> > oldDeltaPs_; ///< Old values of parameters changes (only for momentum learning)
}; // GradientDescentWithMomentumBackpropagationAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_GRADIENT_DESCENT_H
