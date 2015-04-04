#ifndef FL_ANFIS_TRAINING_H
#define FL_ANFIS_TRAINING_H


#include <cstddef>
#include <deque>
#include <fl/anfis/engine.h>
#include <fl/dataset.h>
#include <fl/detail/math.h>
#include <fl/detail/rls.h>
#include <fl/detail/terms.h>
#include <fl/detail/traits.h>
#include <fl/fuzzylite.h>
#include <fl/Headers.h>
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
 * The hybrid learning algorithm has been proposed by J.-S.R. Jang in [1] and is
 * well explained in [2].
 * What follows is an extert of the description of the hybrid algorithm as
 * found in [2].
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
 *
 * References
 * -# J.-S.R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference Systems," IEEE Transactions on Systems, Man, and Cybernetics, 23:3(665-685), 1993.
 * -# J.-S.R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class Jang1993HybridLearningAlgorithm
{
	/**
	 * Constructor
	 *
	 * \param p_anfis Pointer to the ANFIS model to be trained
	 * \param ss The initial step size used in the parameter update formula
	 * \param ssDecrRate The step size decrease rate
	 * \param ssIncrRate The step size increase rate
	 * \param ff The forgetting factor used in the recursive least square algorithm
	 */
    public: explicit Jang1993HybridLearningAlgorithm(Engine* p_anfis,
													 fl::scalar ss = 0.01,
													 fl::scalar ssDecrRate = 0.9,
													 fl::scalar ssIncrRate = 1.1,
													 fl::scalar ff = 1)
	: p_anfis_(p_anfis),
	  stepSizeInit_(ss),
	  stepSizeDecrRate_(ssDecrRate),
	  stepSizeIncrRate_(ssIncrRate),
	  stepSizeErrWindowLen_(5),
	  //useBias_(true),
	  rls_(0,0,0,ff)
    {
		this->init();
	}

	/// Sets the initial step size
	public: void setInitialStepSize(fl::scalar value)
	{
		stepSizeInit_ = value;
	}

	/// Gets the initial step size
	public: fl::scalar getInitialStepSize() const
	{
		return stepSizeInit_;
	}

	/// Sets the step size decrease rate
	public: void setStepSizeDecreaseRate(fl::scalar value)
	{
		stepSizeDecrRate_ = value;
	}

	/// Gets the step size decrease rate
	public: fl::scalar getStepSizeDecreaseRate() const
	{
		return stepSizeDecrRate_;
	}

	/// Sets the step size increase rate
	public: void setStepSizeIncreaseRate(fl::scalar value)
	{
		stepSizeIncrRate_ = value;
	}

	/// Gets the step size increase rate
	public: fl::scalar getStepSizeIncreaseRate() const
	{
		return stepSizeIncrRate_;
	}

	/// Sets the forgetting factor
	public: void setForgettingFactor(fl::scalar value)
	{
		rls_.setForgettingFactor(value);
	}

	/// Gets the forgetting factor
	public: fl::scalar getForgettingFactor() const
	{
		return rls_.getForgettingFactor();
	}

	/**
	 * Trains the ANFIS model
	 *
	 * \param data The training set
	 * \param maxEpochs The maximum number of epochs
	 * \param errorGoal The error to achieve
	 *
	 * The error measure is the Root Mean Squared Error (RMSE).
	 *
	 * \return The achieve error
	 */
	public: fl::scalar train(const fl::DataSet<fl::scalar>& data,
							 std::size_t maxEpochs = 10,
							 fl::scalar errorGoal = 0)
	{
		fl::scalar rmse = 0;
		for (std::size_t epoch = 0; epoch < maxEpochs; ++epoch)
		{
			FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch);

			rmse = this->trainSingleEpoch(data);

			FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch << " -> RMSE: ");

			if (fl::detail::FloatTraits<fl::scalar>::EssentiallyLessEqual(rmse, errorGoal))
			{
				break;
			}
		}
		return rmse;
	}

    /// Trains the ANFIS model for a single epoch only using the given training set \a data
    public: fl::scalar trainSingleEpoch(const fl::DataSet<fl::scalar>& data)
    {
		//const bool oldHasBias = p_anfis_->hasBias();
		//p_anfis_->setHasBias(false);
		p_anfis_->setIsLearning(true);

		// Update parameters of input terms
		if (dEdPs_.size() > 0)
		{
			std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
			const std::size_t ni = fuzzyLayer.size();

			fl::scalar errNorm = 0;
			for (std::size_t i = 0; i < ni; ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];

//std::cerr << "PHASE #-1 - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX
				for (std::size_t p = 0,
								 np = dEdPs_.at(p_node).size();
					 p < np;
					 ++p)
				{
					errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
				}
			}
			errNorm = std::sqrt(errNorm);
std::cerr << "PHASE #-1 - Error Norm: " << errNorm << std::endl;///XXX
std::cerr << "PHASE #-1 - Step Size: " << stepSize_ << std::endl;///XXX
			if (errNorm > 0)
			{
				for (std::size_t i = 0; i < ni; ++i)
				{
					FuzzificationNode* p_node = fuzzyLayer[i];
					std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());

std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
					for (std::size_t p = 0,
									 np = dEdPs_.at(p_node).size();
						 p < np;
						 ++p)
					{
						params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
					}
std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
					detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
				}
			}
		}
		// Update step-size
//std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
//std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
		if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
		{
			if (stepSizeDecrCounter_ == (stepSizeErrWindowLen_-1))
			{
				bool update = true;
				for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
				{
					if (i % 2)
					{
						//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
						update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
					}
					else
					{
						//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
						update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
					}
				}
				if (update)
				{
//std::cerr << "STEP-SIZE (decreasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeDecrRate_) << std::endl;//XXX
					stepSize_ *= stepSizeDecrRate_;
					stepSizeDecrCounter_ = 1;
				}
				else
				{
					++stepSizeDecrCounter_;
				}
			}
			else
			{
				++stepSizeDecrCounter_;
			}
		}
		if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeIncrRate_, 1))
		{
			if (stepSizeIncrCounter_ == (stepSizeErrWindowLen_-1))
			{
				bool update = true;
				for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
				{
					//update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
					update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
				}
				if (update)
				{
//std::cerr << "STEP-SIZE (increasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeIncrRate_) << std::endl;//XXX
					stepSize_ *= stepSizeIncrRate_;
					stepSizeIncrCounter_ = 1;
				}
				else
				{
					++stepSizeIncrCounter_;
				}
			}
			else
			{
				++stepSizeIncrCounter_;
			}
		}

		rls_.reset();
		dEdPs_.clear();
		//if (rlsPhi_.size() > 0)
		//{
		//	// Restore the RLS regressor vector of the previous epoch
		//	rls_.setRegressor(rlsPhi_);
		//}
		//stepSize_ = stepSizeInit_;
		//stepSizeErrWindow_.clear();

		fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch

		// Forwards inputs from input layer to antecedent layer, and estimate parameters with RLS
		//std::vector< std::vector<fl::scalar> > antecedentValues;
		for (typename DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
			 entryIt != entryEndIt;
			 ++entryIt)
		{
			const DataSetEntry<fl::scalar>& entry = *entryIt;

			const std::size_t nout = entry.numOfOutputs();

			if (nout != p_anfis_->numberOfOutputVariables())
			{
				FL_THROW2(std::invalid_argument, "Incorrect output dimension");
			}

			const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

			// Compute current rule firing strengths
			const std::vector<fl::scalar> ruleFiringStrengths = p_anfis_->evalTo(entry.inputBegin(), entry.inputEnd(), fl::anfis::Engine::AntecedentLayer);
			//antecedentValues.push_back(ruleFiringStrengths);

//std::cerr << "PHASE #0 - Traning data #: " << std::distance(data.entryBegin(), entryIt) << std::endl;//XXX
//std::cerr << "PHASE #0 - Entry input: "; fl::detail::VectorOutput(std::cerr, std::vector<fl::scalar>(entry.inputBegin(), entry.inputEnd())); std::cerr << std::endl;//XXX
////std::cerr << "PHASE #0 - Rule firing strength: "; fl::detail::VectorOutput(std::cerr, ruleFiringStrengths); std::cerr << std::endl;//XXX
//{
//	std::vector< std::vector<fl::scalar> > mfParams;
//	std::cerr << "PHASE #0 - Layer 0: [";
//	std::vector<InputNode*> nodes0 = p_anfis_->getInputLayer();
//	for (std::size_t i = 0; i < nodes0.size(); ++i)
//	{
//		std::cerr << nodes0[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 1: [";
//	std::vector<FuzzificationNode*> nodes1 = p_anfis_->getFuzzificationLayer();
//	for (std::size_t i = 0; i < nodes1.size(); ++i)
//	{
//		std::cerr << nodes1[i]->getValue() << " ";
//		mfParams.push_back(detail::GetTermParameters(nodes1[i]->getTerm()));
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 2: ";
//	std::vector<InputHedgeNode*> nodes2 = p_anfis_->getInputHedgeLayer();
//	for (std::size_t i = 0; i < nodes2.size(); ++i)
//	{
//		std::cerr << nodes2[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 3: ";
//	std::vector<AntecedentNode*> nodes3 = p_anfis_->getAntecedentLayer();
//	for (std::size_t i = 0; i < nodes3.size(); ++i)
//	{
//		std::cerr << nodes3[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//
//	for (std::size_t i = 0; i < mfParams.size(); ++i)
//	{
//		std::cerr << "MF #" << i << " Parameters: "; fl::detail::VectorOutput(std::cerr, mfParams[i]); std::cerr << std::endl;
//	}
//}
			// Compute normalization factor
			const fl::scalar totRuleFiringStrength = fl::detail::Sum<fl::scalar>(ruleFiringStrengths.begin(), ruleFiringStrengths.end());
//{//[XXX]
//std::cerr << "PHASE #0 - Total Rule firing strength: " << totRuleFiringStrength << std::endl;
//std::cerr << "PHASE #0 - Normalized Rule firing strength: ";
//for (std::size_t i = 0; i < ruleFiringStrengths.size(); ++i)
//{
//	std::cerr << ruleFiringStrengths[i]/totRuleFiringStrength << " ";
//}
//std::cerr << std::endl;
//}//[XXX]
			// Compute input to RLS algorithm
			std::vector<fl::scalar> rlsInputs(rls_.getInputDimension());
			{
				std::size_t k = 0;
				std::size_t r = 0;
				for (std::size_t v = 0,
								 nv = p_anfis_->numberOfOutputVariables();
					 v < nv;
					 ++v)
				{
					fl::OutputVariable* p_var = p_anfis_->getOutputVariable(v);

					FL_DEBUG_ASSERT( p_var );

					for (std::size_t t = 0,
									 nt = p_var->numberOfTerms();
						 t < nt;
						 ++t)
					{
						fl::Term* p_term = p_var->getTerm(t);

						FL_DEBUG_ASSERT( p_term );

						const std::size_t numParams = detail::GetTermParameters(p_term).size();
						for (std::size_t p = 1; p < numParams; ++p)
						{
							rlsInputs[k] = ruleFiringStrengths[r]*entry.getField(p-1)/totRuleFiringStrength;
							++k;
						}
						rlsInputs[k] = ruleFiringStrengths[r]/totRuleFiringStrength;
						++k;
						++r;
					}
				}
			}
//std::cerr << "PHASE #0 - Num inputs: " << rls_.getInputDimension() << " - Num Outputs: " << rls_.getOutputDimension() << " - Order: " << rls_.getModelOrder() << std::endl;//XXX
//std::cerr << "PHASE #0 - RLS Input: "; fl::detail::VectorOutput(std::cerr, rlsInputs); std::cerr << std::endl;///XXX
			// Estimate parameters
			std::vector<fl::scalar> actualOut;
			actualOut = rls_.estimate(rlsInputs.begin(), rlsInputs.end(), targetOut.begin(), targetOut.end());
//std::cerr << "PHASE #0 - Target: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - Actual: ";fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << std::endl;///XXX
		}

		// Put estimated RLS parameters in the ANFIS model and save RLS regressor vector
		{
			const std::vector< std::vector<fl::scalar> > rlsParamMatrix = rls_.getEstimatedParameters();
//std::cerr << "PHASE #0 - Estimated RLS params: "; fl::detail::MatrixOutput(std::cerr, rlsParamMatrix); std::cerr << std::endl;//XXX

			std::size_t k = 0;
			//std::size_t r = 0;
			for (std::size_t v = 0,
							 nv = p_anfis_->numberOfOutputVariables();
				 v < nv;
				 ++v)
			{
				fl::OutputVariable* p_var = p_anfis_->getOutputVariable(v);

				FL_DEBUG_ASSERT( p_var );

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
std::cerr << "PHASE #0 - Estimated RLS params - Output #" << v << " - Term #" << t << " - Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;//XXX
					//++r;
				}
			}

			//rlsPhi_ = rls_.getRegressor();
		}

/*
		for (std::size_t e = 0,
						 ne = data.size();
			 e < ne;
			 ++e)
		{
			const fl::DataSetEntry<fl::scalar> entry = data.get(e);
			const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

std::cerr << "PHASE #1 - Data entry #" << e << std::endl;//XXX
std::cerr << "PHASE #1 - PHASE #1 - Restore output from layer 0 to 3: ["; //XXX
			// Restores old values up to antecedent layer
			std::vector<AntecedentNode*> antecedentLayer = p_anfis_->getAntecedentLayer();
			for (std::size_t i = 0,
							 ni = antecedentLayer.size();
				 i < ni;
				 ++i)
			{
				AntecedentNode* p_node = antecedentLayer[i];

				p_node->setValue(antecedentValues[e][i]);
std::cerr << p_node->getValue() << " ";
			}
std::cerr << "]" << std::endl;//XXX

			// Forward values from consequent layer to output layer
			std::vector<fl::scalar> actualOut = p_anfis_->evalFrom(fl::anfis::Engine::ConsequentLayer);
*/
		for (typename DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
			 entryIt != entryEndIt;
			 ++entryIt)
		{
			const DataSetEntry<fl::scalar>& entry = *entryIt;
//std::cerr << "PHASE #1 - Traning data #: " << std::distance(data.entryBegin(), entryIt) << std::endl;//XXX

			const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

			// Compute ANFIS output
			const std::vector<fl::scalar> actualOut = p_anfis_->eval(entry.inputBegin(), entry.inputEnd());

			// Update bias in case of zero rule firing strength
			if (p_anfis_->hasBias())
			{
				bool skip = false;

				for (std::size_t i = 0,
								 ni = actualOut.size();
					 i < ni;
					 ++i)
				{
					if (fl::Operation::isNaN(actualOut[i]))
					{
						OutputNode* p_outNode = p_anfis_->getOutputLayer().at(i);

						FL_DEBUG_ASSERT( p_outNode );

						//bias_[i] += stepSize_*(targetOut[i]-bias_[i]);
						fl::scalar bias = p_outNode->getBias();
						bias += stepSize_*(targetOut[i]-bias);
						p_outNode->setBias(bias);
						skip = true;
					}
				}
				//p_anfis_->setBias(bias_);

				if (skip)
				{
					// Skip this data point
//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, bias_); std::cerr << std::endl; //XXX
					continue;
				}
			}

//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, bias_); std::cerr << std::endl; //XXX

			// Update error
			fl::scalar se = 0;
			for (std::size_t i = 0,
							 ni = targetOut.size();
				 i < ni;
				 ++i)
			{
				const fl::scalar out = fl::Operation::isNaN(actualOut[i]) ? 0.0 : actualOut[i];

				se += fl::detail::Sqr(targetOut[i]-out);
			}
			rmse += se;
//std::cerr << "PHASE #1 - Current error: " <<  se << " - Total error: " << rmse << std::endl;//XXX

			// Backward errors
			std::map<const Node*,fl::scalar> dEdOs;
			// Computes error derivatives at output layer
			{
				const std::vector<OutputNode*> outLayer = p_anfis_->getOutputLayer();
				for (std::size_t i = 0,
								 ni = targetOut.size();
					 i < ni;
					 ++i)
				{
					const Node* p_node = outLayer[i];

					dEdOs[p_node] = -2.0*(targetOut[i]-actualOut[i]);
//std::cerr << "PHASE #1 - Layer: " << Engine::OutputLayer << ", Node: " << i << ", dEdO: " << dEdOs[p_node] << std::endl;//XXX
				}
			}
			// Propagates errors back to the fuzzification layer
			for (Engine::LayerCategory layerCat = p_anfis_->getPreviousLayerCategory(Engine::OutputLayer);
				 layerCat != Engine::InputLayer;
				 layerCat = p_anfis_->getPreviousLayerCategory(layerCat))
			{
				std::vector<Node*> layer = p_anfis_->getLayer(layerCat);

				for (std::size_t i = 0,
								 ni = layer.size();
					 i < ni;
					 ++i)
				{
					Node* p_fromNode = layer[i];

					fl::scalar dEdO = 0;
					std::vector<Node*> outConns = p_fromNode->outputConnections();
					for (std::size_t j = 0,
									 nj = outConns.size();
						 j < nj;
						 ++j)
					{
						Node* p_toNode = outConns[j];

						const std::vector<fl::scalar> dOdOs = p_toNode->evalDerivativeWrtInputs();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - dOdOs: "; fl::detail::VectorOutput(std::cerr, dOdOs); std::cerr << std::endl;//XXX

						// Find the index k in the input connection of p_fromNode related to the input node p_toNode
						const std::vector<Node*> inConns = p_toNode->inputConnections();
						const std::size_t nk = inConns.size();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - #In Conns: " << nk << " - double check #In Conns: " << p_anfis_->inputConnections(p_toNode).size() << std::endl;//XXX
						std::size_t k = 0;
						while (k < nk && inConns[k] != p_fromNode)
						{
							++k;
						}
						if (k == nk)
						{
							FL_THROW2(std::runtime_error, "Found inconsistencies in input and output connections");
						}
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", Node: " << i << " - Found k: " << k << std::endl;//XXX

						dEdO += dEdOs[p_toNode]*dOdOs[k];
					}
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", Node: " << i << "..." << std::endl;//XXX
					dEdOs[p_fromNode] = dEdO;
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << ", dEdO: " << dEdOs[p_fromNode] << std::endl;//XXX
				}
			}

//std::cerr << "PHASE #1 - Updating parameters" << std::endl;
			// Update error derivatives wrt parameters $\frac{\partial E}{\partial P_{ij}}$
			std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
			for (std::size_t i = 0,
							 ni = fuzzyLayer.size();
				 i < ni;
				 ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];

				const std::vector<fl::scalar> dOdPs = p_node->evalDerivativeWrtParams();
//std::cerr << "PHASE #1 - Layer: " << Engine::FuzzificationLayer << ", Node: " << i << " (" << p_node << "), dOdPs: "; fl::detail::VectorOutput(std::cerr, dOdPs); std::cerr << std::endl;//XXX
				const std::size_t np = dOdPs.size();

				if (dEdPs_.count(p_node) == 0)
				{
					dEdPs_[p_node].resize(np, 0);
				}

				for (std::size_t p = 0; p < np; ++p)
				{
					dEdPs_[p_node][p] += dEdOs[p_node]*dOdPs[p];
//std::cerr << "PHASE #1 - Layer: fuzzification, Node: " << i << " (" << p_node << "), dEdP_" << p << ": " << dEdPs_.at(p_node)[p] << std::endl;//XXX
				}
			}
		}

		rmse = std::sqrt(rmse/data.size());

		if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
		{
			stepSizeErrWindow_.pop_back();
		}
		stepSizeErrWindow_.push_front(rmse);

#if 0
		// Update parameters of input terms
		if (dEdPs_.size() > 0)
		{
			std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
			const std::size_t ni = fuzzyLayer.size();

			fl::scalar errNorm = 0;
			for (std::size_t i = 0; i < ni; ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];

//std::cerr << "PHASE #2 - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX
				for (std::size_t p = 0,
								 np = dEdPs_.at(p_node).size();
					 p < np;
					 ++p)
				{
					errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
				}
			}
			errNorm = std::sqrt(errNorm);
//std::cerr << "PHASE #2 - Error Norm: " << errNorm << std::endl;///XXX
//std::cerr << "PHASE #2 - Step Size: " << stepSize_ << std::endl;///XXX
			if (errNorm > 0)
			{
				for (std::size_t i = 0; i < ni; ++i)
				{
					FuzzificationNode* p_node = fuzzyLayer[i];
					std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());

std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
					for (std::size_t p = 0,
									 np = dEdPs_.at(p_node).size();
						 p < np;
						 ++p)
					{
						params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
					}
std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
					detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
				}
			}
		}

		// Update step-size
//std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
//std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
		if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
		{
			if (stepSizeDecrCounter_ == (stepSizeErrWindowLen_-1))
			{
				bool update = true;
				for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
				{
					if (i % 2)
					{
						//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
						update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
					}
					else
					{
						//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
						update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
					}
				}
				if (update)
				{
//std::cerr << "STEP-SIZE (decreasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeDecrRate_) << std::endl;//XXX
					stepSize_ *= stepSizeDecrRate_;
					stepSizeDecrCounter_ = 1;
				}
				else
				{
					++stepSizeDecrCounter_;
				}
			}
			else
			{
				++stepSizeDecrCounter_;
			}
		}
		if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeIncrRate_, 1))
		{
			if (stepSizeIncrCounter_ == (stepSizeErrWindowLen_-1))
			{
				bool update = true;
				for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
				{
					//update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
					update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
				}
				if (update)
				{
//std::cerr << "STEP-SIZE (increasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeIncrRate_) << std::endl;//XXX
					stepSize_ *= stepSizeIncrRate_;
					stepSizeIncrCounter_ = 1;
				}
				else
				{
					++stepSizeIncrCounter_;
				}
			}
			else
			{
				++stepSizeIncrCounter_;
			}
		}
#endif // if 0

		//p_anfis_->setHasBias(oldHasBias);
		p_anfis_->setIsLearning(false);

		return rmse;
	}

	/// Initialize the training algorithm
	private: void init()
	{
		std::size_t numParams = 0;
		//std::size_t numOutTerms = 0;
		for (std::size_t i = 0,
						 nv = p_anfis_->numberOfOutputVariables();
			 i < nv;
			 ++i)
		{
			fl::OutputVariable* p_var = p_anfis_->getOutputVariable(i);

			FL_DEBUG_ASSERT( p_var );

			for (std::size_t j = 0,
							 nt = p_var->numberOfTerms();
				 j < nt;
				 ++j)
			{
				fl::Term* p_term = p_var->getTerm(j);

				FL_DEBUG_ASSERT( p_term );

				numParams += detail::GetTermParameters(p_term).size();

				//++numOutTerms;
			}
		}
		//numParams *= numOutTerms;

		rls_.setModelOrder(0);
		rls_.setInputDimension(numParams);
		rls_.setOutputDimension(p_anfis_->numberOfOutputVariables());
		rls_.reset();
		//rlsPhi_.clear();

		dEdPs_.clear();
		stepSize_ = stepSizeInit_;
		stepSizeIncrCounter_ = stepSizeDecrCounter_ = 0;
		stepSizeErrWindow_.clear();

		//bias_.clear();
		//if (useBias_)
		//{
		//	bias_.resize(p_anfis_->numberOfOutputVariables(), 0);
		//}
    }


	private: Engine* p_anfis_; ///< The ANFIS model
	private: fl::scalar stepSizeInit_; ///< The initial value of the step size
	private: fl::scalar stepSizeDecrRate_; ///< The rate at which the step size must be decreased
	private: fl::scalar stepSizeIncrRate_; ///< The rate at which the step size must be increased
	private: fl::scalar stepSize_; ///< Step size to use in the parameter update formula representing the length of each transition along the gradient direction in the parameter space
	private: std::size_t stepSizeErrWindowLen_; ///< Length of the RMSE window used to update the step size
	private: std::deque<fl::scalar> stepSizeErrWindow_; ///< Window of RMSEs used to update the step size
	private: std::size_t stepSizeIncrCounter_; ///< Counter used to check when to increase the step size
	private: std::size_t stepSizeDecrCounter_; ///< Counter used to check when to decrease the step size
	//private: bool useBias_; ///< if \c true, add a bias to handle zero-firing error
	//private: std::vector<fl::scalar> bias_; ///< The bias to use in the output
	private: fl::detail::RecursiveLeastSquaresEstimator<fl::scalar> rls_; ///< The recursive least-squares estimator
	private: std::map< Node*, std::vector<fl::scalar> > dEdPs_; // Error derivatives wrt node parameters
	//private: std::vector<fl::scalar> rlsPhi_; ///< RLS regressor of the last epoch
}; // Jang1993HybridLearningAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_H
