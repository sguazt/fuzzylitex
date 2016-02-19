/**
 * \file fl/anfis/training/jang1993_hybrid.h
 *
 * \brief Hybrid training algorithm by (J.-S-R. Jang,1993)
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

#ifndef FL_ANFIS_TRAINING_JANG1993_HYBRID_H
#define FL_ANFIS_TRAINING_JANG1993_HYBRID_H


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
 * Hybrid learning algorithm by (J.-S.R. Jang,1993)
 *
 * The hybrid learning algorithm uses a combination of the gradient-descent
 * backpropagation algorithm and least-squares estimation to identify the
 * parameters of the input and output membership functions of a single-output,
 * Sugeno-type fuzzy inference system.
 *
 * The hybrid learning algorithm has been proposed by J.-S.R. Jang in [Jang1993]
 * and is well explained in [Jang1997].
 * What follows is an extert of the description of the hybrid algorithm as
 * found in [Jang1997].
 * In the batch mode of the hybrid learning algorithm, each epoch is composed of
 * a <em>forward pass</em> and a <em>backward pass</em>.
 * In the forward pass, after an input vector is presented, the outputs of the
 * nodes in the ANFIS adaptive network are computed layer by layer in order to
 * build a row for matrices \f$A\f$ and \f$y\f$.
 * This process is repeated for all the training data pairs to form a complete
 * \f$A\f$ and \f$y\f$.
 * The the parameters \f$S_2\f$ of the output terms in the rule consequents are
 * identified by a least-squares method (e.g., the recursive least-squares
 * algorithm).
 * After the parameters \f$S_2\f$ are identified, the error measure (i.e., the
 * squared error) can be computed for each training data pair.
 * In the backward pass, the error signals (i.e., the derivative of the error
 * measure with respect to each node output) propagate from the output end
 * toward the input end.
 * The gradient vector is accumulated for each training data entry.
 * At the end of the backward pass for all training data, the parameters
 * \f$S_1\f$ of the input terms are updated according to the steepest descent.
 * For given fixed values of parameters \f$S_1\f$, the parameters \f$S_2\f$ thus
 * found are guaranteed to be the global optimum point in the \f$S_2\f$
 * parameter space because of the choice of the squared error measure.
 *
 * In the backward step, parameters \f$\alpha\f$ are updated according to the
 * generalized delta rule formula (typically used by the backpropagation
 * algorithm):
 * \f{align}
 *   \alpha &= \alpha + \Delta\alpha,\\
 *   \Delta\alpha &= -\eta\frac{\partial E}{\partial \alpha},\\
 *   \eta &= \frac{\kappa}{\sqrt{\sum_\alpha \frac{\partial E}{\partial a}}}
 * \f}
 * where:
 * - \f$\eta\f$ is the learning rate,
 * - \f$\kappa\f$ is the step size, representing the length of each transition
 *   along the gradient direction in the parameter space.
 * - \f$E\f$ is the error measure which is typically the sum of squared errors:
 *   \f[
 *     E=\sum_{k=1}^N (d_k-o_k)^2
 *   \f]
 *   where \f$d_k\f$ is the desired value and \f$o_k\f$ is the actual output.
 * .
 * The <em>step-size</em> represents the length of each transition along the
 * gradient direction in the parameter space and may influence the speed of
 * convergence.
 * Specifically, as discussed in [Jang1993], if \f$\kappa\f$ is small, the
 * gradient method will closely approximate the gradient path, but convergence
 * will be slow since the gradient must be calculated many times.
 * On the other hand, if \f$\kappa\f$ is large, convergence will initially be
 * very fast, but the algorithm will oscillate about the optimum.
 * Based on this observations, the <em>step-size</em> is updated according to
 * the following strategy (as suggested by [Jang1993]):
 * - If the error undergoes four consecutive reductions, increase the step-size
 *   by multiplying it by a constant positive numeric value greater than one.
 * - If the error undergoes two consecutive combinations of one increase and one
 *   reduction, decrease the step-size by multiplying it by a constant positive
 *   numeric value less than one.
 * .
 *
 * References
 * -# [Jang1993] J.-S.R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference Systems," IEEE Transactions on Systems, Man, and Cybernetics, 23:3(665-685), 1993.
 * -# [Jang1997] J.-S.R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API Jang1993HybridLearningAlgorithm: public TrainingAlgorithm
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
    explicit Jang1993HybridLearningAlgorithm(Engine* p_anfis = fl::null,
                                             fl::scalar ss = 0.01,
                                             fl::scalar ssDecrRate = 0.9,
                                             fl::scalar ssIncrRate = 1.1,
                                             fl::scalar ff = 1);

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

    /// Updates the step-size (and the learning rate as well)
    void updateStepSize();

    /// Resets state for single epoch training
    void resetSingleEpoch();

    /// Gets the number of parameters of each output term
    std::size_t numberOfOutputTermParameters() const;

    /// Trains the ANFIS model for a single epoch only using the given training set \a trainData
    fl::scalar doTrainSingleEpoch(const fl::DataSet<fl::scalar>& trainData);

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
    fl::detail::RecursiveLeastSquaresEstimator<fl::scalar> rls_; ///< The recursive least-squares estimator
    //fl::detail::KalmanFilter<fl::scalar> rls_; ///< The recursive least-squares estimator
    std::map< Node*, std::vector<fl::scalar> > dEdPs_; ///< Error derivatives wrt node parameters
    std::map< Node*, std::vector<fl::scalar> > oldDeltaPs_; ///< Old values of parameters changes (only for momentum learning)
}; // Jang1993HybridLearningAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_JANG1993_HYBRID_H
