/**
 * \file fl/anfis/training/gradient_descent.cpp
 *
 * \brief Definitions for the ANFIS training algorithms based on gradient descent backpropagation
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

#include <cmath>
#include <cstddef>
#include <fl/anfis/engine.h>
#include <fl/anfis/training/gradient_descent.h>
#include <fl/dataset.h>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/fuzzylite.h>
#include <fl/Operation.h>
#include <fl/term/Linear.h>
#include <fl/term/Term.h>
#include <fl/variable/OutputVariable.h>
#include <map>
#include <vector>


namespace fl { namespace anfis {

///////////////////////////////////////////////////
// GradientDescentBackpropagationAlgorithm
///////////////////////////////////////////////////


GradientDescentBackpropagationAlgorithm::GradientDescentBackpropagationAlgorithm(Engine* p_anfis)
: BaseType(p_anfis),
  online_(false)
{
	this->init();
}

void GradientDescentBackpropagationAlgorithm::setIsOnline(bool value)
{
	online_ = value;
}

bool GradientDescentBackpropagationAlgorithm::isOnline() const
{
	return online_;
}

void GradientDescentBackpropagationAlgorithm::setCurrentError(fl::scalar value)
{
	curError_ = value;
}

fl::scalar GradientDescentBackpropagationAlgorithm::getCurrentError() const
{
	return curError_;
}

const std::map< Node*, std::vector<fl::scalar> >& GradientDescentBackpropagationAlgorithm::getErrorDerivatives() const
{
	return dEdPs_;
}

fl::scalar GradientDescentBackpropagationAlgorithm::doTrainSingleEpoch(const fl::DataSet<fl::scalar>& data)
{
	this->check();

	fl::scalar rmse = 0;

	if (online_)
	{
		rmse = this->trainSingleEpochOnline(data);
	}
	else
	{
		rmse = this->trainSingleEpochOffline(data);
	}

	//TODO
	//this->setEpochError(rmse);
	//this->setTotalError(this->getTotalError()+rmse);

	return rmse;
}

void GradientDescentBackpropagationAlgorithm::doReset()
{
	this->init();
}

fl::scalar GradientDescentBackpropagationAlgorithm::trainSingleEpochOffline(const fl::DataSet<fl::scalar>& data)
{
	this->resetSingleEpoch();

	fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch

	// Forwards inputs from input layer to the output layer
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

//std::cerr << "PHASE #0 - Traning data #: " << std::distance(data.entryBegin(), entryIt) << std::endl;//XXX
//std::cerr << "PHASE #0 - Entry input: "; fl::detail::VectorOutput(std::cerr, std::vector<fl::scalar>(entry.inputBegin(), entry.inputEnd())); std::cerr << std::endl;//XXX

		// Compute ANFIS output
		const std::vector<fl::scalar> actualOut = this->getEngine()->eval(entry.inputBegin(), entry.inputEnd());

		// Update bias in case of zero rule firing strength
		if (this->getEngine()->hasBias())
		{
			const bool skip = this->updateBias(targetOut, actualOut);

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
		Engine::LayerCategory layerCat = Engine::InputLayer;
		do
		{
			layerCat = this->getEngine()->getNextLayerCategory(layerCat);

			std::vector<Node*> layer = this->getEngine()->getLayer(layerCat);
			for (std::size_t i = 0,
							 ni = layer.size();
				 i < ni;
				 ++i)
			{
				Node* p_node = layer[i];

				const std::vector<fl::scalar> dOdPs = p_node->evalDerivativeWrtParams();
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << " (" << p_node << "), dOdPs: "; fl::detail::VectorOutput(std::cerr, dOdPs); std::cerr << std::endl;//XXX
				const std::size_t np = dOdPs.size();

				if (dEdPs_.count(p_node) == 0)
				{
					dEdPs_[p_node].resize(np, 0);
				}

				for (std::size_t p = 0; p < np; ++p)
				{
					dEdPs_[p_node][p] += dEdOs[p_node]*dOdPs[p];
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << " (" << p_node << "), dEdP_" << p << ": " << dEdPs_.at(p_node)[p] << std::endl;//XXX
				}
			}
		}
		while (layerCat != Engine::OutputLayer);
	}

	rmse = std::sqrt(rmse/data.size());

	this->setCurrentError(rmse);

	// Update parameters of input terms
	this->updateInputParameters();

	return rmse;
}

fl::scalar GradientDescentBackpropagationAlgorithm::trainSingleEpochOnline(const fl::DataSet<fl::scalar>& data)
{
	this->resetSingleEpoch();

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

		// Resets error signals
		dEdPs_.clear();

		// Compute ANFIS output
		const std::vector<fl::scalar> actualOut = this->getEngine()->eval(entry.inputBegin(), entry.inputEnd());

		// Update bias in case of zero rule firing strength
		if (this->getEngine()->hasBias())
		{
			const bool skip = this->updateBias(targetOut, actualOut);

			if (skip)
			{
				// Skip this data point
//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, this->getEngine()->getBias()); std::cerr << std::endl; //XXX
				continue;
			}
		}

std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, this->getEngine()->getBias()); std::cerr << std::endl; //XXX

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
		this->setCurrentError(squaredErr);
std::cerr << "PHASE #1 - Current error: " <<  squaredErr << " - Total error: " << rmse << std::endl;//XXX

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
		Engine::LayerCategory layerCat = Engine::InputLayer;
		do
		{
			layerCat = this->getEngine()->getNextLayerCategory(layerCat);

			std::vector<Node*> layer = this->getEngine()->getLayer(layerCat);
			for (std::size_t i = 0,
							 ni = layer.size();
				 i < ni;
				 ++i)
			{
				Node* p_node = layer[i];

				const std::vector<fl::scalar> dOdPs = p_node->evalDerivativeWrtParams();
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << " (" << p_node << "), dOdPs: "; fl::detail::VectorOutput(std::cerr, dOdPs); std::cerr << std::endl;//XXX
				const std::size_t np = dOdPs.size();

				if (dEdPs_.count(p_node) == 0)
				{
					dEdPs_[p_node].resize(np, 0);
				}

				for (std::size_t p = 0; p < np; ++p)
				{
					dEdPs_[p_node][p] += dEdOs[p_node]*dOdPs[p];
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << " (" << p_node << "), dEdP_" << p << ": " << dEdPs_.at(p_node)[p] << std::endl;//XXX
				}
			}
		}
		while (layerCat != Engine::OutputLayer);

		// Update parameters of input terms
		this->updateInputParameters();
	}

	rmse = std::sqrt(rmse/data.size());

	return rmse;
}

void GradientDescentBackpropagationAlgorithm::updateInputParameters()
{
	this->doUpdateInputParameters();
}

bool GradientDescentBackpropagationAlgorithm::updateBias(const std::vector<fl::scalar>& targetOut, const std::vector<fl::scalar>& actualOut)
{
	return this->doUpdateBias(targetOut, actualOut);
}

void GradientDescentBackpropagationAlgorithm::resetSingleEpoch()
{
	dEdPs_.clear();
	curError_ = 0;

	this->doResetSingleEpoch();
}

void GradientDescentBackpropagationAlgorithm::init()
{
	dEdPs_.clear();
	curError_ = 0;
}

void GradientDescentBackpropagationAlgorithm::check() const
{
	if (this->getEngine() == fl::null)
	{
		FL_THROW2(std::logic_error, "Invalid ANFIS engine");
	}

	this->doCheck();
}


///////////////////////////////////////////////////
// Jang1993GradientDescentBackpropagationAlgorithm
///////////////////////////////////////////////////


Jang1993GradientDescentBackpropagationAlgorithm::Jang1993GradientDescentBackpropagationAlgorithm(Engine* p_anfis,
																								 fl::scalar ss,
																								 fl::scalar ssDecrRate,
																								 fl::scalar ssIncrRate)
: BaseType(p_anfis),
  stepSizeInit_(ss),
  stepSizeDecrRate_(ssDecrRate),
  stepSizeIncrRate_(ssIncrRate),
  stepSizeErrWindowLen_(5),
  stepSizeIncrCounter_(0),
  stepSizeDecrCounter_(0)
{
	this->init();
}

void Jang1993GradientDescentBackpropagationAlgorithm::setInitialStepSize(fl::scalar value)
{
	stepSizeInit_ = fl::detail::FloatTraits<fl::scalar>::DefinitelyMax(value, 0);
}

fl::scalar Jang1993GradientDescentBackpropagationAlgorithm::getInitialStepSize() const
{
	return stepSizeInit_;
}

void Jang1993GradientDescentBackpropagationAlgorithm::setStepSizeDecreaseRate(fl::scalar value)
{
	stepSizeDecrRate_ = fl::detail::FloatTraits<fl::scalar>::DefinitelyMax(value, 0);
}

fl::scalar Jang1993GradientDescentBackpropagationAlgorithm::getStepSizeDecreaseRate() const
{
	return stepSizeDecrRate_;
}

void Jang1993GradientDescentBackpropagationAlgorithm::setStepSizeIncreaseRate(fl::scalar value)
{
	stepSizeIncrRate_ = fl::detail::FloatTraits<fl::scalar>::DefinitelyMax(value, 0);
}

fl::scalar Jang1993GradientDescentBackpropagationAlgorithm::getStepSizeIncreaseRate() const
{
	return stepSizeIncrRate_;
}

void Jang1993GradientDescentBackpropagationAlgorithm::doReset()
{
	BaseType::doReset();
	this->init();
}

void Jang1993GradientDescentBackpropagationAlgorithm::doResetSingleEpoch()
{
}

void Jang1993GradientDescentBackpropagationAlgorithm::doUpdateInputParameters()
{
	const std::map< Node*, std::vector<fl::scalar> >& dEdPs = this->getErrorDerivatives();

	// Update parameters of input terms
	if (dEdPs.size() > 0)
	{
		// Computes the error norm

		fl::scalar errNorm = 0;

		Engine::LayerCategory layerCat = Engine::InputLayer;
		do
		{
			layerCat = this->getEngine()->getNextLayerCategory(layerCat);

			std::vector<Node*> layer = this->getEngine()->getLayer(layerCat);

			for (std::size_t i = 0,
							 ni = layer.size();
				 i < ni;
				 ++i)
			{
				Node* p_node = layer[i];

//std::cerr << "PHASE #-1 - Layer: " << layerCat << " - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX

				for (std::size_t p = 0,
								 np = dEdPs.at(p_node).size();
					 p < np;
					 ++p)
				{
					errNorm += fl::detail::Sqr(dEdPs.at(p_node).at(p));
				}
			}
		}
		while (layerCat != Engine::OutputLayer);

		errNorm = std::sqrt(errNorm);
std::cerr << "PHASE #-1 - Layer: " << layerCat << " - Error Norm: " << errNorm << std::endl;///XXX
std::cerr << "PHASE #-1 - Layer: " << layerCat << " - STEP-SIZE: " << stepSize_ << std::endl;///XXX
std::cerr << "PHASE #-1 - Layer: " << layerCat << " - Learning Rate: " << (stepSize_/errNorm) << std::endl;///XXX

		// Update parameters

		if (errNorm > 0)
		{
			const fl::scalar learningRate = stepSize_/errNorm;

			layerCat = Engine::InputLayer;
			do
			{
				layerCat = this->getEngine()->getNextLayerCategory(layerCat);

				std::vector<Node*> layer = this->getEngine()->getLayer(layerCat);

				for (std::size_t i = 0,
								 ni = layer.size();
					 i < ni;
					 ++i)
				{
					Node* p_node = layer[i];

					// check: null
					FL_DEBUG_ASSERT( p_node );

					const std::size_t np = dEdPs.at(p_node).size();

					std::vector<fl::scalar> params = p_node->getParams();

std::cerr << "PHASE #-1 - Layer: " << layerCat << " - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Layer: " << layerCat << " - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs.at(p_node)); std::cerr << std::endl;///XXX

					for (std::size_t p = 0; p < np; ++p)
					{
						const fl::scalar deltaP = -learningRate*dEdPs.at(p_node).at(p);

						params[p] += deltaP;
					}
std::cerr << "PHASE #-1 - Layer: " << layerCat << " - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
					p_node->setParams(params.begin(), params.end());
				}
			}
			while (layerCat != Engine::OutputLayer);
		}
	}

	// Update step-size
	this->updateStepSize();
}

bool Jang1993GradientDescentBackpropagationAlgorithm::doUpdateBias(const std::vector<fl::scalar>& targetOut, const std::vector<fl::scalar>& actualOut)
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

	return skip;
}

void Jang1993GradientDescentBackpropagationAlgorithm::updateStepSize()
{
	if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
	{
		stepSizeErrWindow_.pop_back();
		//stepSizeErrWindow_.pop_front();
	}
	stepSizeErrWindow_.push_front(this->getCurrentError());
	//stepSizeErrWindow_.push_back(this->getCurrentError());

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

void Jang1993GradientDescentBackpropagationAlgorithm::init()
{
	stepSize_ = stepSizeInit_;
	stepSizeIncrCounter_ = stepSizeDecrCounter_ = 0;
	stepSizeErrWindow_.clear();
}

void Jang1993GradientDescentBackpropagationAlgorithm::doCheck() const
{
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

std::size_t Jang1993GradientDescentBackpropagationAlgorithm::numberOfOutputTermParameters() const
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


///////////////////////////////////////////////////
// GradientDescentBackpropagationAlgorithm
///////////////////////////////////////////////////


GradientDescentWithMomentumBackpropagationAlgorithm::GradientDescentWithMomentumBackpropagationAlgorithm(Engine* p_anfis,
																										 fl::scalar learningRate,
																										 fl::scalar momentum)
: BaseType(p_anfis),
  learnRate_(learningRate),
  momentum_(momentum)
{
	this->init();
}

void GradientDescentWithMomentumBackpropagationAlgorithm::setLearningRate(fl::scalar value)
{
	learnRate_ = fl::detail::FloatTraits<fl::scalar>::DefinitelyMax(value, 0);
}

fl::scalar GradientDescentWithMomentumBackpropagationAlgorithm::getLearningRate() const
{
	return learnRate_;
}

void GradientDescentWithMomentumBackpropagationAlgorithm::setMomentum(fl::scalar value)
{
	momentum_ = fl::detail::FloatTraits<fl::scalar>::Clamp(value, 0, 1);
}

fl::scalar GradientDescentWithMomentumBackpropagationAlgorithm::getMomentum() const
{
	return momentum_;
}

void GradientDescentWithMomentumBackpropagationAlgorithm::doReset()
{
	BaseType::doReset();

	this->init();
}

void GradientDescentWithMomentumBackpropagationAlgorithm::doResetSingleEpoch()
{
	oldDeltaPs_.clear();
}

void GradientDescentWithMomentumBackpropagationAlgorithm::doUpdateInputParameters()
{
	const std::map< Node*, std::vector<fl::scalar> >& dEdPs = this->getErrorDerivatives();

	if (dEdPs.size() > 0)
	{
		Engine::LayerCategory layerCat = Engine::InputLayer;
		do
		{
			layerCat = this->getEngine()->getNextLayerCategory(layerCat);

			std::vector<Node*> layer = this->getEngine()->getLayer(layerCat);

			for (std::size_t i = 0,
							 ni = layer.size();
				 i < ni;
				 ++i)
			{
				Node* p_node = layer[i];

				// check: null
				FL_DEBUG_ASSERT( p_node );

				const std::size_t np = dEdPs.at(p_node).size();

				std::vector<fl::scalar> params = p_node->getParams();

std::cerr << "PHASE #-1 - Layer: " << layerCat << " - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Layer: " << layerCat << " - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs.at(p_node)); std::cerr << std::endl;///XXX
				if (momentum_ > 0 && oldDeltaPs_.count(p_node) == 0)
				{
					oldDeltaPs_[p_node].resize(np, 0);
				}

				for (std::size_t p = 0; p < np; ++p)
				{
					fl::scalar deltaP = -learnRate_*dEdPs.at(p_node).at(p);

					if (momentum_ > 0)
					{
						const fl::scalar oldDeltaP = oldDeltaPs_.at(p_node).at(p);

						deltaP = (1-momentum_)*deltaP + momentum_*oldDeltaP;

						oldDeltaPs_[p_node][p] = deltaP;
					}

					params[p] += deltaP;
				}
std::cerr << "PHASE #-1 - Layer: " << layerCat << " - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
				p_node->setParams(params.begin(), params.end());
			}
		}
		while (layerCat != Engine::OutputLayer);
	}
}

bool GradientDescentWithMomentumBackpropagationAlgorithm::doUpdateBias(const std::vector<fl::scalar>& targetOut, const std::vector<fl::scalar>& actualOut)
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

			//FIXME: should we take into account also the momentum?

			fl::scalar bias = p_outNode->getBias();
			bias += learnRate_*(targetOut[i]-bias);
			p_outNode->setBias(bias);
			skip = true;
		}
	}
	//this->getEngine()->setBias(bias_);

	return skip;
}

void GradientDescentWithMomentumBackpropagationAlgorithm::init()
{
	oldDeltaPs_.clear();
}

void GradientDescentWithMomentumBackpropagationAlgorithm::doCheck() const
{
	if (learnRate_ <= 0)
	{
		FL_THROW2(std::logic_error, "Invalid learning rate");
	}
	if (momentum_ < 0 || momentum_ > 1)
	{
		FL_THROW2(std::logic_error, "Invalid momentum");
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

std::size_t GradientDescentWithMomentumBackpropagationAlgorithm::numberOfOutputTermParameters() const
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
