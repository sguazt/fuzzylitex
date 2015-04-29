/**
 * \file fl/anfis/training/jang1993_hybrid.cpp
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
#include <fl/anfis/training/jang1993_hybrid.h>
#include <fl/dataset.h>
#include <fl/detail/math.h>
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

Jang1993HybridLearningAlgorithm::Jang1993HybridLearningAlgorithm(Engine* p_anfis,
																 fl::scalar ss,
																 fl::scalar ssDecrRate,
																 fl::scalar ssIncrRate,
																 fl::scalar ff)
: BaseType(p_anfis),
  stepSizeInit_(ss),
  stepSizeDecrRate_(ssDecrRate),
  stepSizeIncrRate_(ssIncrRate),
  stepSizeErrWindowLen_(5),
  stepSizeIncrCounter_(0),
  stepSizeDecrCounter_(0),
  online_(false),
  //momentum_(0),
  //useBias_(true),
  rls_(0,0,0,ff)
{
	this->init();
}

void Jang1993HybridLearningAlgorithm::setInitialStepSize(fl::scalar value)
{
	stepSizeInit_ = fl::detail::FloatTraits<fl::scalar>::DefinitelyMax(value, 0);
}

fl::scalar Jang1993HybridLearningAlgorithm::getInitialStepSize() const
{
	return stepSizeInit_;
}

void Jang1993HybridLearningAlgorithm::setStepSizeDecreaseRate(fl::scalar value)
{
	stepSizeDecrRate_ = fl::detail::FloatTraits<fl::scalar>::DefinitelyMax(value, 0);
}

fl::scalar Jang1993HybridLearningAlgorithm::getStepSizeDecreaseRate() const
{
	return stepSizeDecrRate_;
}

void Jang1993HybridLearningAlgorithm::setStepSizeIncreaseRate(fl::scalar value)
{
	stepSizeIncrRate_ = fl::detail::FloatTraits<fl::scalar>::DefinitelyMax(value, 0);
}

fl::scalar Jang1993HybridLearningAlgorithm::getStepSizeIncreaseRate() const
{
	return stepSizeIncrRate_;
}

//void Jang1993HybridLearningAlgorithm::setMomentum(fl::scalar value)
//{
//	momentum_ = fl::detail::FloatTraits<fl::scalar>::Clamp(value, 0, 1);
//}

//fl::scalar Jang1993HybridLearningAlgorithm::getMomentum() const
//{
//	return momentum_;
//}

void Jang1993HybridLearningAlgorithm::setForgettingFactor(fl::scalar value)
{
	rls_.setForgettingFactor(value);
}

fl::scalar Jang1993HybridLearningAlgorithm::getForgettingFactor() const
{
	return rls_.getForgettingFactor();
}

void Jang1993HybridLearningAlgorithm::setIsOnline(bool value)
{
	online_ = value;
}

bool Jang1993HybridLearningAlgorithm::isOnline() const
{
	return online_;
}

fl::scalar Jang1993HybridLearningAlgorithm::doTrainSingleEpoch(const fl::DataSet<fl::scalar>& data)
{
	this->check();

/*
	// Update parameters of input terms
	this->updateInputParameters();

	// Update step-size
	this->updateStepSize();

	rls_.reset();
	dEdPs_.clear();
	//if (rlsPhi_.size() > 0)
	//{
	//	// Restore the RLS regressor vector of the previous epoch
	//	rls_.setRegressor(rlsPhi_);
	//}
	//stepSize_ = stepSizeInit_;
	//stepSizeErrWindow_.clear();
*/

	fl::scalar rmse = 0;

	if (online_)
	{
		rmse = this->trainSingleEpochOnline(data);
	}
	else
	{
		rmse = this->trainSingleEpochOffline(data);
	}

/*
	if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
	{
		stepSizeErrWindow_.pop_back();
		//stepSizeErrWindow_.pop_front();
	}
	stepSizeErrWindow_.push_front(rmse);
	//stepSizeErrWindow_.push_back(rmse);
*/

	return rmse;
}

void Jang1993HybridLearningAlgorithm::doReset()
{
	this->init();
}

fl::scalar Jang1993HybridLearningAlgorithm::trainSingleEpochOffline(const fl::DataSet<fl::scalar>& data)
{
	// Update parameters of input terms
	this->updateInputParameters();

	// Update step-size
	this->updateStepSize();

	//rls_.reset();
	//dEdPs_.clear();
	this->resetSingleEpoch();

	const std::size_t numOutTermParams = this->numberOfOutputTermParameters();

	fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch

	// Forwards inputs from input layer to antecedent layer, and estimate parameters with RLS
	//std::vector< std::vector<fl::scalar> > antecedentValues;
	for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
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

//std::cerr << "PHASE #0 - Traning data #: " << std::distance(data.entryBegin(), entryIt) << std::endl;//XXX
//std::cerr << "PHASE #0 - Entry input: "; fl::detail::VectorOutput(std::cerr, std::vector<fl::scalar>(entry.inputBegin(), entry.inputEnd())); std::cerr << std::endl;//XXX
//std::cerr << "PHASE #0 - Rule firing strength: "; fl::detail::VectorOutput(std::cerr, ruleFiringStrengths); std::cerr << std::endl;//XXX
//{[XXX]
//	std::vector< std::vector<fl::scalar> > mfParams;
//	std::cerr << "PHASE #0 - Layer 0: [";
//	std::vector<InputNode*> nodes0 = this->getEngine()->getInputLayer();
//	for (std::size_t i = 0; i < nodes0.size(); ++i)
//	{
//		std::cerr << nodes0[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 1: [";
//	std::vector<FuzzificationNode*> nodes1 = this->getEngine()->getFuzzificationLayer();
//	for (std::size_t i = 0; i < nodes1.size(); ++i)
//	{
//		std::cerr << nodes1[i]->getValue() << " ";
//		mfParams.push_back(detail::GetTermParameters(nodes1[i]->getTerm()));
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 2: ";
//	std::vector<InputHedgeNode*> nodes2 = this->getEngine()->getInputHedgeLayer();
//	for (std::size_t i = 0; i < nodes2.size(); ++i)
//	{
//		std::cerr << nodes2[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 3: ";
//	std::vector<AntecedentNode*> nodes3 = this->getEngine()->getAntecedentLayer();
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
//}[/XXX]
		// Compute input to RLS algorithm
		std::vector<fl::scalar> rlsInputs(rls_.getInputDimension());
		{
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
	}

	// Put estimated RLS parameters in the ANFIS model
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

	for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
		 entryIt != entryEndIt;
		 ++entryIt)
	{
		const fl::DataSetEntry<fl::scalar>& entry = *entryIt;
//std::cerr << "PHASE #1 - Traning data #: " << std::distance(data.entryBegin(), entryIt) << std::endl;//XXX

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

					//bias_[i] += stepSize_*(targetOut[i]-bias_[i]);
					fl::scalar bias = p_outNode->getBias();
					bias += stepSize_*(targetOut[i]-bias);
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
//std::cerr << "PHASE #1 - Current error: " <<  squaredErr << " - Total error: " << rmse << std::endl;//XXX

		// Backward errors
		std::map<const Node*,fl::scalar> dEdOs;
		// Computes error derivatives at output layer
		{
			const std::vector<OutputNode*> outLayer = this->getEngine()->getOutputLayer();
			for (std::size_t i = 0,
							 ni = targetOut.size();
				 i < ni;
				 ++i)
			{
				const Node* p_node = outLayer[i];
				const fl::scalar out = fl::Operation::isNaN(actualOut[i]) ? 0.0 : actualOut[i];

				dEdOs[p_node] = -2.0*(targetOut[i]-out);
//std::cerr << "PHASE #1 - Layer: " << Engine::OutputLayer << ", Node: " << i << ", dEdO: " << dEdOs[p_node] << std::endl;//XXX
			}
		}
		// Propagates errors back to the fuzzification layer
		for (Engine::LayerCategory layerCat = this->getEngine()->getPreviousLayerCategory(Engine::OutputLayer);
			 layerCat != Engine::InputLayer;
			 layerCat = this->getEngine()->getPreviousLayerCategory(layerCat))
		{
			std::vector<Node*> layer = this->getEngine()->getLayer(layerCat);

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
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - #In Conns: " << nk << " - double check #In Conns: " << this->getEngine()->inputConnections(p_toNode).size() << std::endl;//XXX
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

		// Update error derivatives wrt parameters $\frac{\partial E}{\partial P_{ij}}$
		std::vector<FuzzificationNode*> fuzzyLayer = this->getEngine()->getFuzzificationLayer();
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
		//stepSizeErrWindow_.pop_front();
	}
	stepSizeErrWindow_.push_front(rmse);
	//stepSizeErrWindow_.push_back(rmse);

	return rmse;
}

fl::scalar Jang1993HybridLearningAlgorithm::trainSingleEpochOnline(const fl::DataSet<fl::scalar>& data)
{
	//rls_.reset();
	this->resetSingleEpoch();

	const std::size_t numOutTermParams = this->numberOfOutputTermParameters();

	fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch

	// Forwards inputs from input layer to antecedent layer, and estimate parameters with RLS
	//std::vector< std::vector<fl::scalar> > antecedentValues;
	for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
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

		// Update parameters of input terms
		this->updateInputParameters();

		// Update step-size
		this->updateStepSize();

		// Resets error signals
		dEdPs_.clear();

		// Compute current rule firing strengths
		const std::vector<fl::scalar> ruleFiringStrengths = this->getEngine()->evalTo(entry.inputBegin(), entry.inputEnd(), fl::anfis::Engine::AntecedentLayer);

		// Compute input to RLS algorithm
		std::vector<fl::scalar> rlsInputs(rls_.getInputDimension());
		{
			// Compute normalization factor
			const fl::scalar totRuleFiringStrength = fl::detail::Sum<fl::scalar>(ruleFiringStrengths.begin(), ruleFiringStrengths.end());

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

		// Put estimated RLS parameters in the ANFIS model
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

					//bias_[i] += stepSize_*(targetOut[i]-bias_[i]);
					fl::scalar bias = p_outNode->getBias();
					bias += stepSize_*(targetOut[i]-bias);
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
//std::cerr << "PHASE #1 - Current error: " <<  squaredErr << " - Total error: " << rmse << std::endl;//XXX

		// Backward errors
		std::map<const Node*,fl::scalar> dEdOs;
		// Computes error derivatives (i.e., error signals) at output layer
		{
			const std::vector<OutputNode*> outLayer = this->getEngine()->getOutputLayer();
			for (std::size_t i = 0,
							 ni = targetOut.size();
				 i < ni;
				 ++i)
			{
				const Node* p_node = outLayer[i];
				const fl::scalar out = fl::Operation::isNaN(actualOut[i]) ? 0.0 : actualOut[i];

				dEdOs[p_node] = -2.0*(targetOut[i]-out);
//std::cerr << "PHASE #1 - Layer: " << Engine::OutputLayer << ", Node: " << i << ", dEdO: " << dEdOs[p_node] << std::endl;//XXX
			}
		}
		// Propagates errors back to the fuzzification layer and computes error signals at each layer
		for (Engine::LayerCategory layerCat = this->getEngine()->getPreviousLayerCategory(Engine::OutputLayer);
			 layerCat != Engine::InputLayer;
			 layerCat = this->getEngine()->getPreviousLayerCategory(layerCat))
		{
			std::vector<Node*> layer = this->getEngine()->getLayer(layerCat);

			for (std::size_t i = 0,
							 ni = layer.size();
				 i < ni;
				 ++i)
			{
				Node* p_fromNode = layer[i];

				// Computes the error signal dE/dO by means of the chain rule:
				//  $\frac{\partial E_p}{\partial x_{l,i}} = \sum_{m=1}^{N(l+1)} \frac{\partial E_p}{\partial x_{l+1,m}}\frac{\partial f_{l+1,m}}{\partial x_{l,i}}$
				fl::scalar dEdO = 0; // This will hold $\frac{\partial E_p}{\partial x_{l,i}}$
				std::vector<Node*> outConns = p_fromNode->outputConnections();
				for (std::size_t j = 0,
								 nj = outConns.size();
					 j < nj;
					 ++j)
				{
					Node* p_toNode = outConns[j];

					// Computes $\frac{\partial f_{l+1,m}}{\partial x_{l,i}}$
					const std::vector<fl::scalar> dOdOs = p_toNode->evalDerivativeWrtInputs();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - dOdOs: "; fl::detail::VectorOutput(std::cerr, dOdOs); std::cerr << std::endl;//XXX

					// Find the index k in the input connection of p_fromNode related to the input node p_toNode
					const std::vector<Node*> inConns = p_toNode->inputConnections();
					const std::size_t nk = inConns.size();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - #In Conns: " << nk << " - double check #In Conns: " << this->getEngine()->inputConnections(p_toNode).size() << std::endl;//XXX
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

					// Computes $\frac{\partial E_p}{\partial x_{l+1,m}}\frac{\partial f_{l+1,m}}{\partial x_{l,i}}$ and sum up to the dEdO of current node
					dEdO += dEdOs[p_toNode]*dOdOs[k];
				}
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", Node: " << i << "..." << std::endl;//XXX
				dEdOs[p_fromNode] = dEdO;
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << ", dEdO: " << dEdOs[p_fromNode] << std::endl;//XXX
			}
		}

//std::cerr << "PHASE #1 - Updating parameters" << std::endl;
		// Update error derivatives wrt parameters $\frac{\partial E}{\partial P_{ij}}$
		std::vector<FuzzificationNode*> fuzzyLayer = this->getEngine()->getFuzzificationLayer();
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

//[FIXME]
		if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
		{
			stepSizeErrWindow_.pop_back();
			//stepSizeErrWindow_.pop_front();
		}
		stepSizeErrWindow_.push_front(squaredErr);
		//stepSizeErrWindow_.push_back(squaredErr);
//[/FIXME]
	}

	rmse = std::sqrt(rmse/data.size());

	return rmse;
}

void Jang1993HybridLearningAlgorithm::updateInputParameters()
{
	// Update parameters of input terms
	if (dEdPs_.size() > 0)
	{
		std::vector<FuzzificationNode*> fuzzyLayer = this->getEngine()->getFuzzificationLayer();
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
//std::cerr << "PHASE #-1 - Error Norm: " << errNorm << std::endl;///XXX
//std::cerr << "PHASE #-1 - STEP-SIZE: " << stepSize_ << std::endl;///XXX
		if (errNorm > 0)
		{
			const fl::scalar learningRate = stepSize_/errNorm;

			for (std::size_t i = 0; i < ni; ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];

				// check: null
				FL_DEBUG_ASSERT( p_node );

				const std::size_t np = dEdPs_.at(p_node).size();

				std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());

//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
				//if (momentum_ > 0 && oldDeltaPs_.count(p_node) == 0)
				//{
				//	oldDeltaPs_[p_node].resize(np, 0);
				//}

				for (std::size_t p = 0; p < np; ++p)
				{
					const fl::scalar deltaP = -learningRate*dEdPs_.at(p_node).at(p);

					params[p] += deltaP;

					//if (momentum_ > 0)
					//{
					//	const fl::scalar oldDeltaP = oldDeltaPs_.at(p_node).at(p);
					//
					//	params[p] += momentum_*oldDeltaP;
					//
					//	oldDeltaPs_[p_node][p] = deltaP;
					//}
				}
//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
				detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
			}
		}
	}
}

void Jang1993HybridLearningAlgorithm::updateStepSize()
{
//std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
//std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
	{
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeDecrCounter_ >= maxCounter)
		{
//std::cerr << "STEP-SIZE decrease checking..." << std::endl;//XXX
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				if ((i % 2) != 0)
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] <= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
				}
				else
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
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
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeIncrCounter_ >= maxCounter)
		{
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				////update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
				//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
				if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
				{
					update = false;
				}
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
}

void Jang1993HybridLearningAlgorithm::resetSingleEpoch()
{
	rls_.reset();
	dEdPs_.clear();
	oldDeltaPs_.clear();
}

void Jang1993HybridLearningAlgorithm::init()
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

	dEdPs_.clear();
	stepSize_ = stepSizeInit_;
	stepSizeIncrCounter_ = stepSizeDecrCounter_ = 0;
	stepSizeErrWindow_.clear();
	oldDeltaPs_.clear();

	//bias_.clear();
	//if (useBias_)
	//{
	//	bias_.resize(this->getEngine()->numberOfOutputVariables(), 0);
	//}
}

void Jang1993HybridLearningAlgorithm::check() const
{
	if (this->getEngine() == fl::null)
	{
		FL_THROW2(std::logic_error, "Invalid ANFIS engine");
	}
	if (this->getEngine()->type() != fl::Engine::TakagiSugeno)
	{
		FL_THROW2(std::logic_error, "This learning algorithm currently works only for Takagi-Sugeno ANFIS");
	}
	if (stepSizeInit_ <= 0)
	{
		FL_THROW2(std::logic_error, "Invalid step-size");
	}
	if (stepSizeDecrRate_ <= 0)
	{
		FL_THROW2(std::logic_error, "Invalid step-size decreasing rate");
	}
	if (stepSizeIncrRate_ <= 0)
	{
		FL_THROW2(std::logic_error, "Invalid step-size increasing rate");
	}
	if (stepSizeErrWindowLen_ == 0)
	{
		FL_THROW2(std::logic_error, "Invalid length for the step-size error window");
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

std::size_t Jang1993HybridLearningAlgorithm::numberOfOutputTermParameters() const
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
