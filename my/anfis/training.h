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

	if (dynamic_cast<const fl::Concave*>(p_term))
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

	std::vector<fl::scalar> params(first, last);

	if (dynamic_cast<const fl::Concave*>(p_term))
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
	  bias_(true),
	  rls_(0,0,0,ff)
    {
		this->init();
	}

    /// Training for a single epoch
    public: fl::scalar trainSingleEpoch(const DataSet<fl::scalar>& data)
    {
		rls_.reset();
		dEdPs_.clear();
		//stepSize_ = stepSizeInit_;
		//stepSizeErrWindow_.clear();

		fl::scalar totErr = 0;

		std::vector< std::vector<fl::scalar> > antecedentValues;
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
			antecedentValues.push_back(ruleFiringStrengths);

std::cerr << "Entry input: "; fl::detail::VectorOutput(std::cerr, std::vector<fl::scalar>(entry.inputBegin(), entry.inputEnd())); std::cerr << std::endl;//XXX
//std::cerr << "Rule firing strength: "; fl::detail::VectorOutput(std::cerr, ruleFiringStrengths); std::cerr << std::endl;//XXX
//{
//	std::cerr << "Layer 0: [";
//	std::vector<InputNode*> nodes0 = p_anfis_->getInputLayer();
//	for (std::size_t i = 0; i < nodes0.size(); ++i)
//	{
//		std::cerr << nodes0[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "Layer 1: [";
//	std::vector<FuzzificationNode*> nodes1 = p_anfis_->getFuzzificationLayer();
//	for (std::size_t i = 0; i < nodes1.size(); ++i)
//	{
//		std::cerr << nodes1[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "Layer 2: ";
//	std::vector<InputHedgeNode*> nodes2 = p_anfis_->getInputHedgeLayer();
//	for (std::size_t i = 0; i < nodes2.size(); ++i)
//	{
//		std::cerr << nodes2[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "Layer 3: ";
//	std::vector<AntecedentNode*> nodes3 = p_anfis_->getAntecedentLayer();
//	for (std::size_t i = 0; i < nodes3.size(); ++i)
//	{
//		std::cerr << nodes3[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//}
			// Compute normalization factor
			const fl::scalar totRuleFiringStrength = fl::detail::Sum<fl::scalar>(ruleFiringStrengths.begin(), ruleFiringStrengths.end());
//{//[XXX]
//std::cerr << "Total Rule firing strength: " << totRuleFiringStrength << std::endl;
//std::cerr << "Normalized Rule firing strength: ";
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
std::cerr << "Num inputs: " << rls_.getInputDimension() << " - Num Outputs: " << rls_.getOutputDimension() << " - Order: " << rls_.getModelOrder() << std::endl;//XXX
std::cerr << "RLS Input: "; fl::detail::VectorOutput(std::cerr, rlsInputs); std::cerr << std::endl;///XXX
			// Estimate parameters
			std::vector<fl::scalar> actualOut;
			actualOut = rls_.estimate(rlsInputs.begin(), rlsInputs.end(), targetOut.begin(), targetOut.end());
std::cerr << "Target: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - Actual: ";fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << std::endl;///XXX
		}

		// Put estimated parameters in the ANFIS model
		{
			const std::vector< std::vector<fl::scalar> > rlsParamMatrix = rls_.getEstimatedParameters();
std::cerr << "Estimated RLS params: "; fl::detail::MatrixOutput(std::cerr, rlsParamMatrix); std::cerr << std::endl;//XXX
			std::size_t k = 0;
			std::size_t r = 0;
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

					const std::size_t numParams = detail::GetTermParameters(p_term).size();
					std::vector<fl::scalar> params(numParams);
					for (std::size_t p = 0; p < numParams; ++p)
					{
						params[p] = rlsParamMatrix[k][i];
						++k;
					}
					detail::SetTermParameters(p_term, params.begin(), params.end());
					++r;
				}
			}
		}

		for (std::size_t e = 0,
						 ne = data.size();
			 e < ne;
			 ++e)
		{
std::cerr << "Data entr #" << e << std::endl;//XXX
			// Restores old values up to antecedent layer
			std::vector<AntecedentNode*> antecedentLayer = p_anfis_->getAntecedentLayer();
			for (std::size_t i = 0; i < antecedentLayer.size(); ++i)
			{
				AntecedentNode* p_node = antecedentLayer[i];
				p_node->setValue(antecedentValues[e][i]);
			}

			// Forward values from consequent layer
			std::vector<fl::scalar> actualOut = p_anfis_->evalFrom(fl::anfis::Engine::ConsequentLayer);

//			if (fis->skipdatapoint)
//			{
//			fis->bias[0] = fis->bias[0] + (fis->ss * (fis->trn_data[j][fis->in_n] - fis->bias[0]) );
//				continue;
//			}

			const fl::DataSetEntry<fl::scalar> entry = data.get(e);
			const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());
std::cerr << "Target output: " << targetOut.back() << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << std::endl; //XXX

			// Update error
			fl::scalar se = 0;
			for (std::size_t i = 0,
							 ni = targetOut.size();
				 i < ni;
				 ++i)
			{
				se += fl::detail::Sqr(targetOut[i]-actualOut[i]);
			}
			totErr += se;
std::cerr << "Current error: " <<  se << " - Total error: " << totErr << std::endl;//XXX

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
	std::cerr << "Layer: " << Engine::OutputLayer << ", Node: " << i << ", dEdO: " << dEdOs[p_node] << std::endl;//XXX
				}
			}
			// Propagates errors back to the fuzzification layer
			for (Engine::LayerCategory layerCat = p_anfis_->getPreviousLayerCategory(Engine::OutputLayer);
				 layerCat != Engine::FuzzificationLayer;
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
std::cerr << "Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - dOdOs: "; fl::detail::VectorOutput(std::cerr, dOdOs); std::cerr << std::endl;//XXX

						// Find the index k in the input connection of p_fromNode related to the input node p_toNode
						const std::vector<Node*> inConns = p_toNode->inputConnections();
						const std::size_t nk = inConns.size();
std::cerr << "Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - #In Conns: " << nk << " - double check #In Conns: " << p_anfis_->inputConnections(p_toNode).size() << std::endl;//XXX
						std::size_t k = 0;
						while (k < nk && inConns[k] != p_fromNode)
						{
							++k;
						}
						if (k == nk)
						{
							FL_THROW2(std::runtime_error, "Found inconsistencies in input and output connections");
						}
std::cerr << "Computing dEdO for Layer: " << layerCat << ", Node: " << i << " - Found k: " << k << std::endl;//XXX

						dEdO += dEdOs[p_toNode]*dOdOs[k];
					}
std::cerr << "Computing dEdO for Layer: " << layerCat << ", Node: " << i << "..." << std::endl;//XXX
					dEdOs[p_fromNode] = dEdO;
std::cerr << "Layer: " << layerCat << ", Node: " << i << ", dEdO: " << dEdOs[p_fromNode] << std::endl;//XXX
				}
			}

//			/* update de_dp of layer 1*/
std::cerr << "Updating parameters" << std::endl;
			// Update error derivatives wrt parameters $\frac{\partial E}{\partial P_{ij}}$
			std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
			for (std::size_t i = 0,
							 ni = fuzzyLayer.size();
				 i < ni;
				 ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];

				const std::vector<fl::scalar> dOdPs = p_node->evalDerivativeWrtParams();
				const std::size_t np = dOdPs.size();

				if (dEdPs_.count(p_node) == 0)
				{
					dEdPs_[p_node].resize(np, 0);
				}

				for (std::size_t p = 0; p < np; ++p)
				{
					dEdPs_[p_node][p] += dEdOs[p_node]*dOdPs[p];
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

				for (std::size_t p = 0,
								 np = dEdPs_.at(p_node).size();
					 p < np;
					 ++p)
				{
					errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
				}
			}
			errNorm = std::sqrt(errNorm);
			if (errNorm > 0)
			{
				for (std::size_t i = 0; i < ni; ++i)
				{
					FuzzificationNode* p_node = fuzzyLayer[i];
					std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());

std::cerr << "Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
					for (std::size_t p = 0,
									 np = dEdPs_.at(p_node).size();
						 p < np;
						 ++p)
					{
						params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
					}
std::cerr << "Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
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

		dEdPs_.clear();
		stepSize_ = stepSizeInit_;
		stepSizeIncrCounter_ = stepSizeDecrCounter_ = 0;
		stepSizeErrWindow_.clear();
    }


	private: Engine* p_anfis_;
	private: fl::scalar stepSizeInit_;
	private: fl::scalar stepSizeDecrRate_;
	private: fl::scalar stepSizeIncrRate_;
	private: bool bias_;
	private: fl::detail::RecursiveLeastSquaresEstimator<fl::scalar> rls_;
	private: std::map< Node*, std::vector<fl::scalar> > dEdPs_; // Error derivatives wrt node parameters
	private: fl::scalar stepSize_;
	private: std::size_t stepSizeErrWindowLen_;
	private: std::deque<fl::scalar> stepSizeErrWindow_;
	private: std::size_t stepSizeIncrCounter_;
	private: std::size_t stepSizeDecrCounter_;
}; // Jang1993HybridLearningAlgorithm

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_H
