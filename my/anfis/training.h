#ifndef FL_ANFIS_TRAINING_H
#define FL_ANFIS_TRAINING_H


//#include <boost/noncopyable.hpp>
#include <cstddef>
#include <deque>
#include <fl/anfis/engine.h>
#include <fl/fuzzylite.h>
#include <fl/Headers.h>
#include <fl/dataset.h>
#include <fl/detail/math.h>
#include <fl/detail/rls.h>
#include <fl/detail/traits.h>
#include <map>
#include <vector>


namespace fl { namespace anfis {

namespace detail {

std::vector<fl::scalar> GetTermParameters(const fl::Term* p_term)
{
	//FIXME: it would be a good idea to add a pure virtual method in fl::Term
	//       class that returns the vector of parameters, like:
	//         virtual std::vector<fl::scalar> getParameters() = 0;

	std::vector<fl::scalar> params;

	if (dynamic_cast<const fl::Bell*>(p_term))
	{
		const fl::Bell* p_realTerm = dynamic_cast<const fl::Bell*>(p_term);
		params.push_back(p_realTerm->getCenter());
		params.push_back(p_realTerm->getWidth());
		params.push_back(p_realTerm->getSlope());
	}
	else if (dynamic_cast<const fl::Concave*>(p_term))
	{
		const fl::Concave* p_realTerm = dynamic_cast<const fl::Concave*>(p_term);
		params.push_back(p_realTerm->getInflection());
		params.push_back(p_realTerm->getEnd());
	}
	else if (dynamic_cast<const fl::Constant*>(p_term))
	{
		const fl::Constant* p_realTerm = dynamic_cast<const fl::Constant*>(p_term);
		params.push_back(p_realTerm->getValue());
	}
	else if (dynamic_cast<const fl::Cosine*>(p_term))
	{
		const fl::Cosine* p_realTerm = dynamic_cast<const fl::Cosine*>(p_term);
		params.push_back(p_realTerm->getCenter());
		params.push_back(p_realTerm->getWidth());
	}
	else if (dynamic_cast<const fl::Discrete*>(p_term))
	{
		const fl::Discrete* p_realTerm = dynamic_cast<const fl::Discrete*>(p_term);
		const std::vector<fl::Discrete::Pair> pairs = p_realTerm->xy();
		const std::size_t np = pairs.size();
		for (std::size_t p = 0; p < np; ++p)
		{
			params.push_back(pairs[p].first);
			params.push_back(pairs[p].second);
		}
	}
	else if (dynamic_cast<const fl::Linear*>(p_term))
	{
		const fl::Linear* p_realTerm = dynamic_cast<const fl::Linear*>(p_term);
		params = p_realTerm->coefficients();
	}
	if (dynamic_cast<const fl::Ramp*>(p_term))
	{
		const fl::Ramp* p_realTerm = dynamic_cast<const fl::Ramp*>(p_term);
		params.push_back(p_realTerm->getStart());
		params.push_back(p_realTerm->getEnd());
	}
	if (dynamic_cast<const fl::Sigmoid*>(p_term))
	{
		const fl::Sigmoid* p_realTerm = dynamic_cast<const fl::Sigmoid*>(p_term);
		params.push_back(p_realTerm->getInflection());
		params.push_back(p_realTerm->getSlope());
	}
	else if (dynamic_cast<const fl::SShape*>(p_term))
	{
		const fl::SShape* p_realTerm = dynamic_cast<const fl::SShape*>(p_term);
		params.push_back(p_realTerm->getStart());
		params.push_back(p_realTerm->getEnd());
	}
	else if (dynamic_cast<const fl::Triangle*>(p_term))
	{
		const fl::Triangle* p_realTerm = dynamic_cast<const fl::Triangle*>(p_term);
		params.push_back(p_realTerm->getVertexA());
		params.push_back(p_realTerm->getVertexB());
		params.push_back(p_realTerm->getVertexC());
	}
	else if (dynamic_cast<const fl::ZShape*>(p_term))
	{
		const fl::ZShape* p_realTerm = dynamic_cast<const fl::ZShape*>(p_term);
		params.push_back(p_realTerm->getStart());
		params.push_back(p_realTerm->getEnd());
	}

	return params;
}

template <typename IterT>
void SetTermParameters(fl::Term* p_term, IterT first, IterT last)
{
	//FIXME: it would be a good idea to add a pure virtual method in fl::Term
	//       class that returns the vector of parameters, like:
	//         virtual std::vector<fl::scalar> getParameters() = 0;

	const std::vector<fl::scalar> params(first, last);

	if (dynamic_cast<fl::Bell*>(p_term))
	{
		fl::Bell* p_realTerm = dynamic_cast<fl::Bell*>(p_term);
		p_realTerm->setCenter(params[0]);
		p_realTerm->setWidth(params[1]);
		p_realTerm->setSlope(params[2]);
	}
	else if (dynamic_cast<fl::Concave*>(p_term))
	{
		fl::Concave* p_realTerm = dynamic_cast<fl::Concave*>(p_term);
		p_realTerm->setInflection(params[0]);
		p_realTerm->setEnd(params[1]);
	}
	else if (dynamic_cast<fl::Constant*>(p_term))
	{
		fl::Constant* p_realTerm = dynamic_cast<fl::Constant*>(p_term);
		p_realTerm->setValue(params[0]);
	}
	else if (dynamic_cast<fl::Cosine*>(p_term))
	{
		fl::Cosine* p_realTerm = dynamic_cast<fl::Cosine*>(p_term);
		p_realTerm->setCenter(params[0]);
		p_realTerm->setWidth(params[1]);
	}
	else if (dynamic_cast<fl::Discrete*>(p_term))
	{
		fl::Discrete* p_realTerm = dynamic_cast<fl::Discrete*>(p_term);
		const std::size_t np = params.size();
		std::vector<fl::Discrete::Pair> pairs;
		for (std::size_t p = 0; p < (np-1); p += 2)
		{
			pairs.push_back(fl::Discrete::Pair(params[p], params[p+1]));
		}
		p_realTerm->setXY(pairs);
	}
	else if (dynamic_cast<fl::Linear*>(p_term))
	{
		fl::Linear* p_realTerm = dynamic_cast<fl::Linear*>(p_term);
		p_realTerm->setCoefficients(params);
	}
	if (dynamic_cast<fl::Ramp*>(p_term))
	{
		fl::Ramp* p_realTerm = dynamic_cast<fl::Ramp*>(p_term);
		p_realTerm->setStart(params[0]);
		p_realTerm->setEnd(params[1]);
	}
	if (dynamic_cast<fl::Sigmoid*>(p_term))
	{
		fl::Sigmoid* p_realTerm = dynamic_cast<fl::Sigmoid*>(p_term);
		p_realTerm->setInflection(params[0]);
		p_realTerm->setSlope(params[1]);
	}
	else if (dynamic_cast<fl::SShape*>(p_term))
	{
		fl::SShape* p_realTerm = dynamic_cast<fl::SShape*>(p_term);
		p_realTerm->setStart(params[0]);
		p_realTerm->setEnd(params[1]);
	}
	else if (dynamic_cast<fl::Triangle*>(p_term))
	{
		fl::Triangle* p_realTerm = dynamic_cast<fl::Triangle*>(p_term);
		p_realTerm->setVertexA(params[0]);
		p_realTerm->setVertexB(params[1]);
		p_realTerm->setVertexC(params[2]);
	}
	else if (dynamic_cast<fl::ZShape*>(p_term))
	{
		fl::ZShape* p_realTerm = dynamic_cast<fl::ZShape*>(p_term);
		p_realTerm->setStart(params[0]);
		p_realTerm->setEnd(params[1]);
	}
}

} // Namespace detail


/**
 * Hybrid learning algorithm using Gradient-Descent and Least-Squares Estimation by (J.-S.R. Jang,1993)
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
    public: explicit Jang1993HybridLearningAlgorithm(Engine* p_anfis, fl::scalar ff = 1)
	: p_anfis_(p_anfis),
	  stepSizeInit_(0.01),
	  stepSizeDecrRate_(0.9),
	  stepSizeIncrRate_(1.1),
	  stepSizeErrWindowLen_(5),
	  useBias_(true),
	  rls_(0,0,0,ff)
    {
		this->init();
	}

    /// Training for a single epoch
    public: fl::scalar trainSingleEpoch(const DataSet<fl::scalar>& data)
    {
		rls_.reset();
		dEdPs_.clear();
		//if (rlsPhi_.size() > 0)
		//{
		//	// Restore the RLS regressor vector of the previous epoch
		//	rls_.setRegressor(rlsPhi_);
		//}
		//stepSize_ = stepSizeInit_;
		//stepSizeErrWindow_.clear();

		fl::scalar totErr = 0;

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
					std::vector<fl::scalar> params(numParams);
					for (std::size_t p = 0; p < numParams; ++p)
					{
						params[p] = rlsParamMatrix[k][v];
						++k;
					}
					detail::SetTermParameters(p_term, params.begin(), params.end());
					++r;
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
			if (useBias_)
			{
				bool skip = false;

				for (std::size_t i = 0,
								 ni = actualOut.size();
					 i < ni;
					 ++i)
				{
					if (fl::Operation::isNaN(actualOut[i]))
					{
						bias_[i] += stepSize_*(targetOut[i]-bias_[i]);
						skip = true;
					}
				}

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
			totErr += se;
//std::cerr << "PHASE #1 - Current error: " <<  se << " - Total error: " << totErr << std::endl;//XXX

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

		totErr /= data.size();
		totErr = std::sqrt(totErr);

		if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
		{
			stepSizeErrWindow_.pop_front();
		}
		stepSizeErrWindow_.push_back(totErr);

		// Update parameters of input terms
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

//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
					for (std::size_t p = 0,
									 np = dEdPs_.at(p_node).size();
						 p < np;
						 ++p)
					{
						params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
					}
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
					detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
				}
			}
		}

		// Update step-size
		if (stepSizeIncrRate_ != 1)
		{
			if (stepSizeIncrCounter_ == (stepSizeErrWindowLen_-1))
			{
				bool update = true;
				for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
				{
					update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
				}
				if (update)
				{
					stepSize_ *= stepSizeIncrRate_;
				}

				stepSizeIncrCounter_ = 0;
			}
			else
			{
				++stepSizeIncrCounter_;
			}
		}
		if (stepSizeDecrRate_ != 1)
		{
			if (stepSizeDecrCounter_ == 4)
			{
				bool update = true;
				for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
				{
					if (i % 2)
					{
						update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					}
					else
					{
						update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					}
				}

				if (update)
				{
					stepSize_ *= stepSizeDecrRate_;
				}
				stepSizeDecrCounter_ = 0;
			}
			else
			{
				++stepSizeDecrCounter_;
			}
		}

		return totErr;
	}

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

		if (useBias_)
		{
			bias_.resize(p_anfis_->numberOfOutputVariables(), 0);
		}
    }


	private: Engine* p_anfis_;
	private: fl::scalar stepSizeInit_;
	private: fl::scalar stepSizeDecrRate_;
	private: fl::scalar stepSizeIncrRate_;
	private: fl::scalar stepSize_;
	private: std::size_t stepSizeErrWindowLen_;
	private: std::deque<fl::scalar> stepSizeErrWindow_;
	private: std::size_t stepSizeIncrCounter_;
	private: std::size_t stepSizeDecrCounter_;
	private: bool useBias_;
	private: std::vector<fl::scalar> bias_;
	private: fl::detail::RecursiveLeastSquaresEstimator<fl::scalar> rls_;
	private: std::map< Node*, std::vector<fl::scalar> > dEdPs_; // Error derivatives wrt node parameters
	//private: std::vector<fl::scalar> rlsPhi_; ///< RLS regressor of the last epoch
}; // Jang1993HybridLearningAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_H
