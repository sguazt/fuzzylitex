#ifndef FL_ANFIS_TRAINING_H
#define FL_ANFIS_TRAINING_H


//#include <boost/noncopyable.hpp>
#include <cstddef>
#include <fl/anfis/engine.h>
#include <fl/fuzzylite.h>
#include <fl/Headers.h>
#include <fl/dataset.h>
#include <fl/detail/math.h>
#include <fl/detail/rls.h>
#include <fl/detail/traits.h>
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
	  rls_(0,0,0,ff)
    {
		this->init();
	}

    /// Training for a single epoch
    public: void trainSingleEpoch(const DataSet<fl::scalar>& data)
    {
		rls_.reset();

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
		std::vector<fl::scalar> out = p_anfis_->evalFrom(fl::anfis::Engine::ConsequentLayer);
std::cerr << "ANFIS output: "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl; //XXX
std::abort();//XXX
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
    }


	private: Engine* p_anfis_;
	private: fl::detail::RecursiveLeastSquaresEstimator<fl::scalar> rls_;
}; // Jang1993HybridLearningAlgorithm

#if 0


---------------------------------
//FL_DEBUG_TRACE("Feedforward step");//XXX
            // 1. Feedforward pass: propagates the input pattern to the network
            //    and computes the output
            const std::vector<ValueT> actualOut = p_nnet->process(entry.inputBegin(), entry.inputEnd());
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "neuron (" << *(p_nnet->getOutputLayer()->neuronBegin()) << ") - net input: " << (*(p_nnet->getOutputLayer()->neuronBegin()))->getNetInput() << ")" << std::endl;//XXX
//std::cerr << "Actual Output: [";
//std::copy(actualOut.begin(), actualOut.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//#endif // FL_DEBUG
////[/XXX]

//FL_DEBUG_TRACE("Backforward step");//XXX
            // 2. Backward pass: propagates the errors backward through the
            //    network
            this->backpropagateErrors(targetOut, actualOut);

			if (online_)
			{
//FL_DEBUG_TRACE("Update weight step");//XXX
				// 2.3. Update network weights
				this->updateWeightsOnline(targetOut, actualOut);
			}
			else
			{
				this->updateErrorGradients(targetOut, actualOut);
			}

//FL_DEBUG_TRACE("Update error step");//XXX
			// 3. Update total network error
			p_errFunc->update(targetOut.begin(), targetOut.end(), actualOut.begin(), actualOut.end());
//[XXX]
#ifdef FL_DEBUG
std::cerr << "  Weights & Biases: ";
DumpWeightsAndBiases(std::cerr, *(this->getNetwork()));
std::cerr << std::endl;
std::cerr << "  Net Inputs: ";
DumpNetInputs(std::cerr, *(this->getNetwork()));
std::cerr << std::endl;
std::cerr << "  Activations: ";
DumpActivations(std::cerr, *(this->getNetwork()));
std::cerr << std::endl;
#endif // FL_DEBUG
//[/XXX]
        }

		if (!online_)
		{
			this->updateWeightsOffline();
		}
    }

	private: void forwardsInputs(std::vector<fl::scalar> const& inputs)
	{
		for (std::size_t i = 0; i < 
	}

    private: void backpropagateErrors(const std::vector<ValueT>& targetOut, const std::vector<ValueT>& actualOut)
    {
		Network<ValueT>* p_nnet = this->getNetwork();

		// check: null
		FL_DEBUG_ASSERT( p_nnet );

		ErrorFunction<ValueT>* p_errFunc = this->getErrorFunction();

		// check: null
		FL_DEBUG_ASSERT( p_errFunc );

        // 2.1. For each output unit, computes its error term $\frac{\partial E}{\partial \mathrm{net}_j}$ by means of the chain rule:
        //  \begin{equation}
        //   \frac{\partial E}{\partial \mathrm{net}_j} = \frac{\partial E}{\partial o_j}\frac{\partial o_j}{\partial \mathrm{net}_j}
        //  \end{equation}
        //  where:
        //  - $E$ is the error function,
        //  - $\mathrm{net}_j$ is the net input of the unit $j$ computed by means of the net input function (e.g., $\mathrm{net}_j=\sum_i w_{ij}x_{ij}$, the weighted sum of inputs for unti \f$j\f$),
        //  - $o_j$ is the output of unit $j$ computed by means of the activation function (e.g., $\frac{1}{1-e^{x_{\mathrm{net}_j}}}$, the sigmoid function)
        //  .
        {
            Layer<ValueT>* p_outLayer =  this->getNetwork()->getOutputLayer();

            //check: null
            FL_DEBUG_ASSERT( p_outLayer );

            // Computes $\frac{\partial E}{\partial o_j}
            const std::vector<ValueT> dEdOs = p_errFunc->evalDerivativeWrtOutput(targetOut.begin(), targetOut.end(), actualOut.begin(), actualOut.end());
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "Output error terms dEdOs: [";
//std::copy(dEdOs.begin(), dEdOs.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//#endif // FL_DEBUG
////[/XXX]

            std::size_t j = 0;
            for (typename Layer<ValueT>::NeuronIterator neurIt = p_outLayer->neuronBegin(),
                                                        neurEndIt = p_outLayer->neuronEnd();
                 neurIt != neurEndIt;
                 ++neurIt)
            {
                Neuron<ValueT>* p_neuron = *neurIt;

                //check: null
                FL_DEBUG_ASSERT( p_neuron );

				// Gets $\frac{\partial E}{\partial o_j}$
				const ValueT dEdOj = dEdOs[j];

                // Computes $\frac{\partial o_j}{\partial \mathrm{net}_j}$
                ActivationFunction<ValueT>* p_actFunc = p_neuron->getActivationFunction();
                const ValueT dOjdNetj = p_actFunc->evalDerivative(p_neuron->getNetInput());

                // Updates the error term (aka, sensitivity) of this neuron
                //p_neuron->setError(dEdOj*dOjdNetj);
				sensitivies_[p_neuron] = dEdOj*dOjdNetj;
//////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "Output neuron (" << p_neuron << ") - dEdOj: " << dEdOj << " - dOjdNetj: " << dOjdNetj << " -> Error: " << sensitivies_.at(p_neuron) << std::endl;
//#endif // FL_DEBUG
//[/XXX]

                ++j;
            }
        }

        // 2.2. For each hidden layer and hidden unit, computes its error term $\frac{\partial E}{\partial \mathrm{net}_j}$ by means of the chain rule:
        //  \begin{equation}
        //  \begin{aligned}
        //   \frac{\partial E}{\partial \mathrm{net}_j} &= \sum_{k \in O_{j}} \frac{\partial E}{\partial \mathrm{net}_k}\frac{\partial \mathrm{net}_k}{\partial \mathrm{net}_j}
        //                                              &= \sum_{k \in O_{j}} \frac{\partial E}{\partial \mathrm{net}_k}\frac{\partial \mathrm{net}_k}{\partial o_j}\frac{\partial o_j}{\partial \mathrm{net}_j}
        //  \end{aligned}
        //  \end{equation}
        //  where:
        //  - $E$ is the error function,
        //  - $O_j$ is the set of units that have unit $j$ as input (i.e., put in another way, is the output unit connections of unit $j$),
        //  - $\mathrm{net}_j$ is the net input of the unit $j$ computed by means of the net input function (e.g., $\mathrm{net}_j=\sum_i w_{ij}x_{ij}$, the weighted sum of inputs for unti $j$),
        //  - $o_j$ is the output of unit $j$ computed by means of the activation function (e.g., $\frac{1}{1-e^{x_{\mathrm{net}_j}}}$, the sigmoid function)
        //  .
        {
            for (typename Network<ValueT>::ReverseLayerIterator hidLayerIt = p_nnet->hiddenLayerRBegin(),
                                                                      hidLayerEndIt = p_nnet->hiddenLayerREnd();
                 hidLayerIt != hidLayerEndIt;
                 ++hidLayerIt)
            {
                Layer<ValueT>* p_layer = *hidLayerIt;

                //check: null
                FL_DEBUG_ASSERT( p_layer );

                 // For each unit j in this hidden layer, computes the error
				std::size_t j = 0;
                for (typename Layer<ValueT>::NeuronIterator neurIt = p_layer->neuronBegin(),
                                                            neurEndIt = p_layer->neuronEnd();
                     neurIt != neurEndIt;
                     ++neurIt)
                {
                    Neuron<ValueT>* p_neuron = *neurIt;

                    //check: null
                    FL_DEBUG_ASSERT( p_neuron );

                    // Computes $\frac{\partial o_j}{\partial \mathrm{net}_j}$
                    ActivationFunction<ValueT>* p_actFunc = p_neuron->getActivationFunction();
                    const ValueT dOjdNetj = p_actFunc->evalDerivative(p_neuron->getNetInput());

                    // Sums the errors from the neurons of next layer
                    ValueT dEdNetkdNetkdOjs = 0;
					//std::size_t k = 0;
					const std::vector<Connection<ValueT>*> outConns = p_neuron->outputConnections();
                    for (typename std::vector<Connection<ValueT>*>::const_iterator connIt = outConns.begin(),
																				   connEndIt = outConns.end();
                         connIt != connEndIt;
                         ++connIt)
                    {
                        const Connection<ValueT>* p_conn = *connIt;

                        //check: null
                        FL_DEBUG_ASSERT( p_conn );

                        Neuron<ValueT>* p_toNeuron = p_conn->getToNeuron();

                        //check: null
                        FL_DEBUG_ASSERT( p_toNeuron );

                        // Compute $\frac{\partial E}{\partial \mathrm{net}_k}$
                        //const ValueT dEdNetk = p_toNeuron->getError();
                        const ValueT dEdNetk = sensitivies_.at(p_toNeuron);

                        // Compute $\frac{\partial \mathrm{net}_k}{\partial o_j}$
                        NetInputFunction<ValueT>* p_netInFunc = p_toNeuron->getNetInputFunction();
                        std::vector<ValueT> inputs = p_toNeuron->inputs();
                        std::vector<ValueT> weights = p_toNeuron->weights();
                        const std::vector<ValueT> dNetkdOs = p_netInFunc->evalDerivativeWrtInput(inputs.begin(), inputs.end(), weights.begin(), weights.end());

                        dEdNetkdNetkdOjs += dEdNetk*dNetkdOs[j];

						//++k;
                    }

                    // Computes $\frac{\partial E}{\partial \mathrm{net}_j}$
                    const ValueT dEdNetj = dOjdNetj*dEdNetkdNetkdOjs;

                    //p_neuron->setError(dEdNetj);
                    sensitivies_[p_neuron] = dEdNetj;
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "Hidden neuron (" << p_neuron << ") - dOjdNetj: " << dOjdNetj << " - dEdNetkdNetkdOjs: " << dEdNetkdNetkdOjs << " -> Error: " << sensitivies_.at(p_neuron) << std::endl;
//#endif // FL_DEBUG
////[/XXX]

					++j;
                }
            }
        }
    }

    private: void updateWeightsOnline(const std::vector<ValueT>& targetOut, const std::vector<ValueT>& actualOut)
    {
		Network<ValueT>* p_nnet = this->getNetwork();

		// check: null
		FL_DEBUG_ASSERT( p_nnet );

		ErrorFunction<ValueT>* p_errFunc = this->getErrorFunction();

		// check: null
		FL_DEBUG_ASSERT( p_errFunc );

        const ValueT eta = this->getLearningRate();

        // 2.3.2. Update weights in the hidden layers
        {
            for (typename Network<ValueT>::LayerIterator hidLayerIt = p_nnet->hiddenLayerBegin(),
                                                               		  hidLayerEndIt = p_nnet->hiddenLayerEnd();
                 hidLayerIt != hidLayerEndIt;
                 ++hidLayerIt)
            {
                Layer<ValueT>* p_layer = *hidLayerIt;

                //check: null
                FL_DEBUG_ASSERT( p_layer );

                 // For each unit in this hidden layer, computes the error
                for (typename Layer<ValueT>::NeuronIterator neurIt = p_layer->neuronBegin(),
                                                            neurEndIt = p_layer->neuronEnd();
                     neurIt != neurEndIt;
                     ++neurIt)
                {
                    Neuron<ValueT>* p_neuron = *neurIt;

                    //check: null
                    FL_DEBUG_ASSERT( p_neuron );

                    // Get the error term $\frac{\partial E}{\partial \mathrm{net}_j}$
                    //const ValueT dEdNetj = p_neuron->getError();
                    const ValueT dEdNetj = sensitivies_.at(p_neuron);

                    // Computes $\frac{\partial \mathrm{net}_j}{\partial w_{ij}}$
                    NetInputFunction<ValueT>* p_netInFunc = p_neuron->getNetInputFunction();
                    std::vector<ValueT> inputs = p_neuron->inputs();
                    std::vector<ValueT> weights = p_neuron->weights();
					if (p_neuron->hasBias())
					{
						inputs.push_back(1);
						weights.push_back(p_neuron->getBias());
					}
                    const std::vector<ValueT> dNetjdWs = p_netInFunc->evalDerivativeWrtWeight(inputs.begin(), inputs.end(), weights.begin(), weights.end());
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "hidden neuron (" << p_neuron << ") dEdNetj: " << dEdNetj << std::endl;
//std::cerr << "hidden neuron (" << p_neuron << ") inputs: [";
//std::copy(inputs.begin(), inputs.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//std::cerr << "hidden neuron (" << p_neuron << ") weights: [";
//std::copy(weights.begin(), weights.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//std::cerr << "hidden neuron (" << p_neuron << ") dNetjdWs: [";
//std::copy(dNetjdWs.begin(), dNetjdWs.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//#endif // FL_DEBUG
////[/XXX]

                    // For each incoming connection $i$, updates its weights $w_{ij}$ as
                    // \[
                    //  w_{ij} = w_{ij} + \Delta w_{ij}
                    // \]
                    // where $\Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}}
					std::size_t i = 0;
					std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
					for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																			 connEndIt = inConns.end();
                         connIt != connEndIt;
                         ++connIt)
                    {
                        Connection<ValueT>* p_conn = *connIt;

                        FL_DEBUG_ASSERT( p_conn );

                        // Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
                        const ValueT dNetjdWij = dNetjdWs[i];
                        const ValueT deltaWij = -eta*dEdNetj*dNetjdWij;

////[XXX]
//#ifdef FL_DEBUG
//const ValueT oldWeight=p_conn->getWeight();//XXX
//#endif // FL_DEBUG
////[/XXX]
                        p_conn->setWeight(p_conn->getWeight() + deltaWij);
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "hidden neuron (" << p_neuron << ") - connection (" << p_conn->getFromNeuron() << " -> " << p_conn->getToNeuron() << ") - Old Weight: " << oldWeight << " - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_conn->getWeight() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]

						if (momentum_ > 0)
						{
							const ValueT oldDeltaWij = oldDeltaWs_.count(p_conn) > 0 ? oldDeltaWs_.at(p_conn) : 0;

							p_conn->setWeight(p_conn->getWeight() + oldDeltaWij);

							oldDeltaWs_[p_conn] = deltaWij;
						}

						++i;
                    }
					if (p_neuron->hasBias())
					{
                        // Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
                        const ValueT dNetjdWij = dNetjdWs[i];
                        const ValueT deltaWij = -eta*dEdNetj*dNetjdWij;

////[XXX]
//#ifdef FL_DEBUG
//const ValueT oldBias=p_neuron->getBias();//XXX
//#endif // FL_DEBUG
////[/XXX]
						p_neuron->setBias(p_neuron->getBias() + deltaWij);

						if (momentum_ > 0)
						{
							const ValueT oldDeltaWij = oldDeltaBs_.count(p_neuron) > 0 ? oldDeltaBs_.at(p_neuron) : 0;

							p_neuron->setBias(p_neuron->getBias() + oldDeltaWij);

							oldDeltaBs_[p_neuron] = deltaWij;
						}
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "hidden neuron (" << p_neuron << ") - BIAS - Old Weight: " << oldBias << " - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_neuron->getBias() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]
					}
                }
            }
        }

        // 2.3.1. Update weights in the output layer
        {
            Layer<ValueT>* p_outLayer =  this->getNetwork()->getOutputLayer();

            //check: null
            FL_DEBUG_ASSERT( p_outLayer );

            // Computes $\frac{\partial E}{\partial o_j}
            const std::vector<ValueT> dEdOs = p_errFunc->evalDerivativeWrtOutput(targetOut.begin(), targetOut.end(), actualOut.begin(), actualOut.end());
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "output neuron - dEdOs: [";
//std::copy(dEdOs.begin(), dEdOs.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//#endif // FL_DEBUG
////[/XXX]

            for (typename Layer<ValueT>::NeuronIterator neurIt = p_outLayer->neuronBegin(),
                                                        neurEndIt = p_outLayer->neuronEnd();
                 neurIt != neurEndIt;
                 ++neurIt)
            {
                Neuron<ValueT>* p_neuron = *neurIt;

                //check: null
                FL_DEBUG_ASSERT( p_neuron );

                // Get the error term $\frac{\partial E}{\partial \mathrm{net}_j}$
                //const ValueT dEdNetj = p_neuron->getError();
                const ValueT dEdNetj = sensitivies_.at(p_neuron);

                // Computes $\frac{\partial \mathrm{net}_j}{\partial w_{ij}}$
                NetInputFunction<ValueT>* p_netInFunc = p_neuron->getNetInputFunction();
                std::vector<ValueT> inputs = p_neuron->inputs();
                std::vector<ValueT> weights = p_neuron->weights();
				if (p_neuron->hasBias())
				{
					inputs.push_back(1);
					weights.push_back(p_neuron->getBias());
				}
                const std::vector<ValueT> dNetjdWs = p_netInFunc->evalDerivativeWrtWeight(inputs.begin(), inputs.end(), weights.begin(), weights.end());
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "output neuron (" << p_neuron << ") dEdNetj: " << dEdNetj << std::endl;
//std::cerr << "output neuron (" << p_neuron << ") inputs&bias: [";
//std::copy(inputs.begin(), inputs.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//std::cerr << "output neuron (" << p_neuron << ") weights&bias: [";
//std::copy(weights.begin(), weights.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//std::cerr << "output neuron (" << p_neuron << ") dNetjdWs: [";
//std::copy(dNetjdWs.begin(), dNetjdWs.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//#endif // FL_DEBUG
////[/XXX]

                // For each incoming connection $i$, updates its weights $w_{ij}$ as
                // \[
                //  w_{ij} = w_{ij} + \Delta w_{ij}
                // \]
                // where $\Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}}
				std::size_t i = 0;
				std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
                for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																		 connEndIt = inConns.end();
                     connIt != connEndIt;
                     ++connIt)
                {
                    Connection<ValueT>* p_conn = *connIt;

                    FL_DEBUG_ASSERT( p_conn );

                    // Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
                    const ValueT dNetjdWij = dNetjdWs[i];
                    const ValueT deltaWij = -eta*dEdNetj*dNetjdWij;

                    p_conn->setWeight(p_conn->getWeight() + deltaWij);
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "output neuron (" << p_neuron << ") - connection (" << p_conn->getFromNeuron() << " -> " << p_conn->getToNeuron() << ") - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_conn->getWeight() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]

					++i;
                }
				if (p_neuron->hasBias())
				{
					// Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
					const ValueT dNetjdWij = dNetjdWs[i];
					const ValueT deltaWij = -eta*dEdNetj*dNetjdWij;

////[XXX]
//#ifdef FL_DEBUG
//const ValueT oldBias=p_neuron->getBias();//XXX
//#endif // FL_DEBUG
////[/XXX]
					p_neuron->setBias(p_neuron->getBias() + deltaWij);

					if (momentum_ > 0)
					{
						const ValueT oldDeltaWij = oldDeltaBs_.count(p_neuron) > 0 ? oldDeltaBs_.at(p_neuron) : 0;

						p_neuron->setBias(p_neuron->getBias() + oldDeltaWij);

						oldDeltaBs_[p_neuron] = deltaWij;
					}
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "output neuron (" << p_neuron << ") - BIAS - Old Weight: " << oldBias << " - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_neuron->getBias() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]
				}
            }
        }
    }

    private: void updateWeightsOffline()
    {
		Network<ValueT>* p_nnet = this->getNetwork();

		// check: null
		FL_DEBUG_ASSERT( p_nnet );

        const ValueT eta = this->getLearningRate();

        // 2.3.2. Update weights in the hidden layers
        {
            for (typename Network<ValueT>::LayerIterator hidLayerIt = p_nnet->hiddenLayerBegin(),
                                                               		  hidLayerEndIt = p_nnet->hiddenLayerEnd();
                 hidLayerIt != hidLayerEndIt;
                 ++hidLayerIt)
            {
                Layer<ValueT>* p_layer = *hidLayerIt;

                //check: null
                FL_DEBUG_ASSERT( p_layer );

                 // For each unit in this hidden layer, computes the error
                for (typename Layer<ValueT>::NeuronIterator neurIt = p_layer->neuronBegin(),
                                                            neurEndIt = p_layer->neuronEnd();
                     neurIt != neurEndIt;
                     ++neurIt)
                {
                    Neuron<ValueT>* p_neuron = *neurIt;

                    //check: null
                    FL_DEBUG_ASSERT( p_neuron );

                    // For each incoming connection $i$, updates its weights $w_{ij}$ as
                    // \[
                    //  w_{ij} = w_{ij} + \Delta w_{ij}
                    // \]
                    // where $\Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}}
					std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
					for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																			 connEndIt = inConns.end();
                         connIt != connEndIt;
                         ++connIt)
                    {
                        Connection<ValueT>* p_conn = *connIt;

                        FL_DEBUG_ASSERT( p_conn );

                        // Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
                        const ValueT deltaWij = -eta*dEdWs_.at(p_conn);

////[XXX]
//#ifdef FL_DEBUG
//const ValueT oldWeight=p_conn->getWeight();//XXX
//#endif // FL_DEBUG
////[/XXX]
                        p_conn->setWeight(p_conn->getWeight() + deltaWij);
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "hidden neuron (" << p_neuron << ") - connection (" << p_conn->getFromNeuron() << " -> " << p_conn->getToNeuron() << ") - Old Weight: " << oldWeight << " - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_conn->getWeight() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]

						if (momentum_ > 0)
						{
							const ValueT oldDeltaWij = oldDeltaWs_.count(p_conn) > 0 ? oldDeltaWs_.at(p_conn) : 0;

							p_conn->setWeight(p_conn->getWeight() + oldDeltaWij);

							oldDeltaWs_[p_conn] = deltaWij;
						}
                    }
					if (p_neuron->hasBias())
					{
                        // Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
                        const ValueT deltaWij = -eta*dEdBs_.at(p_neuron);

////[XXX]
//#ifdef FL_DEBUG
//const ValueT oldBias=p_neuron->getBias();//XXX
//#endif // FL_DEBUG
////[/XXX]
						p_neuron->setBias(p_neuron->getBias() + deltaWij);

						if (momentum_ > 0)
						{
							const ValueT oldDeltaWij = oldDeltaBs_.count(p_neuron) > 0 ? oldDeltaBs_.at(p_neuron) : 0;

							p_neuron->setBias(p_neuron->getBias() + oldDeltaWij);

							oldDeltaBs_[p_neuron] = deltaWij;
						}
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "hidden neuron (" << p_neuron << ") - BIAS - Old Weight: " << oldBias << " - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_neuron->getBias() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]
					}
                }
            }
        }

        // 2.3.1. Update weights in the output layer
        {
            Layer<ValueT>* p_outLayer =  this->getNetwork()->getOutputLayer();

            //check: null
            FL_DEBUG_ASSERT( p_outLayer );

            for (typename Layer<ValueT>::NeuronIterator neurIt = p_outLayer->neuronBegin(),
                                                        neurEndIt = p_outLayer->neuronEnd();
                 neurIt != neurEndIt;
                 ++neurIt)
            {
                Neuron<ValueT>* p_neuron = *neurIt;

                //check: null
                FL_DEBUG_ASSERT( p_neuron );

                // For each incoming connection $i$, updates its weights $w_{ij}$ as
                // \[
                //  w_{ij} = w_{ij} + \Delta w_{ij}
                // \]
                // where $\Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}}
				std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
                for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																		 connEndIt = inConns.end();
                     connIt != connEndIt;
                     ++connIt)
                {
                    Connection<ValueT>* p_conn = *connIt;

                    FL_DEBUG_ASSERT( p_conn );

                    // Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
                    const ValueT deltaWij = -eta*dEdWs_.at(p_conn);

                    p_conn->setWeight(p_conn->getWeight() + deltaWij);
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "output neuron (" << p_neuron << ") - connection (" << p_conn->getFromNeuron() << " -> " << p_conn->getToNeuron() << ") - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_conn->getWeight() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]
                }
				if (p_neuron->hasBias())
				{
					// Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
					const ValueT deltaWij = -eta*dEdBs_.at(p_neuron);

////[XXX]
//#ifdef FL_DEBUG
//const ValueT oldBias=p_neuron->getBias();//XXX
//#endif // FL_DEBUG
////[/XXX]
					p_neuron->setBias(p_neuron->getBias() + deltaWij);

					if (momentum_ > 0)
					{
						const ValueT oldDeltaWij = oldDeltaBs_.count(p_neuron) > 0 ? oldDeltaBs_.at(p_neuron) : 0;

						p_neuron->setBias(p_neuron->getBias() + oldDeltaWij);

						oldDeltaBs_[p_neuron] = deltaWij;
					}
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "output neuron (" << p_neuron << ") - BIAS - Old Weight: " << oldBias << " - dNetjdWij: " << dNetjdWij << " - DeltaWij: " << deltaWij << " - Weight: " << p_neuron->getBias() << std::endl;///XXX
//#endif // FL_DEBUG
////[/XXX]
				}
            }
        }
    }

    private: void updateErrorGradients(const std::vector<ValueT>& targetOut, const std::vector<ValueT>& actualOut)
    {
		Network<ValueT>* p_nnet = this->getNetwork();
		ErrorFunction<ValueT>* p_errFunc = this->getErrorFunction();

        // 2.3.2. Update error gradients in the hidden layers
        {
            for (typename Network<ValueT>::LayerIterator hidLayerIt = p_nnet->hiddenLayerBegin(),
                                                               		  hidLayerEndIt = p_nnet->hiddenLayerEnd();
                 hidLayerIt != hidLayerEndIt;
                 ++hidLayerIt)
            {
                Layer<ValueT>* p_layer = *hidLayerIt;

                //check: null
                FL_DEBUG_ASSERT( p_layer );

                 // For each unit in this hidden layer, computes the error
                for (typename Layer<ValueT>::NeuronIterator neurIt = p_layer->neuronBegin(),
                                                            neurEndIt = p_layer->neuronEnd();
                     neurIt != neurEndIt;
                     ++neurIt)
                {
                    Neuron<ValueT>* p_neuron = *neurIt;

                    //check: null
                    FL_DEBUG_ASSERT( p_neuron );

                    // Get the error term $\frac{\partial E}{\partial \mathrm{net}_j}$
                    //const ValueT dEdNetj = p_neuron->getError();
                    const ValueT dEdNetj = sensitivies_.at(p_neuron);

                    // Computes $\frac{\partial \mathrm{net}_j}{\partial w_{ij}}$
                    NetInputFunction<ValueT>* p_netInFunc = p_neuron->getNetInputFunction();
                    std::vector<ValueT> inputs = p_neuron->inputs();
                    std::vector<ValueT> weights = p_neuron->weights();
					if (p_neuron->hasBias())
					{
						inputs.push_back(1);
						weights.push_back(p_neuron->getBias());
					}
                    const std::vector<ValueT> dNetjdWs = p_netInFunc->evalDerivativeWrtWeight(inputs.begin(), inputs.end(), weights.begin(), weights.end());

                    // For each incoming connection $i$, updates the error gradient term \frac{\partial E}{\partial w_{ij}}
					std::size_t i = 0;
					std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
					for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																			 connEndIt = inConns.end();
                         connIt != connEndIt;
                         ++connIt)
                    {
                        Connection<ValueT>* p_conn = *connIt;

                        FL_DEBUG_ASSERT( p_conn );

                        const ValueT dNetjdWij = dNetjdWs[i];

                        dEdWs_[p_conn] = dEdNetj*dNetjdWij;

						++i;
                    }
					if (p_neuron->hasBias())
					{
                        const ValueT dNetjdWij = dNetjdWs[i];

                        dEdBs_[p_neuron] = dEdNetj*dNetjdWij;
					}
                }
            }
        }

        // 2.3.1. Update error gradients in the output layer
        {
            Layer<ValueT>* p_outLayer =  this->getNetwork()->getOutputLayer();

            //check: null
            FL_DEBUG_ASSERT( p_outLayer );

            // Computes $\frac{\partial E}{\partial o_j}
            const std::vector<ValueT> dEdOs = p_errFunc->evalDerivativeWrtOutput(targetOut.begin(), targetOut.end(), actualOut.begin(), actualOut.end());

            for (typename Layer<ValueT>::NeuronIterator neurIt = p_outLayer->neuronBegin(),
                                                        neurEndIt = p_outLayer->neuronEnd();
                 neurIt != neurEndIt;
                 ++neurIt)
            {
                Neuron<ValueT>* p_neuron = *neurIt;

                //check: null
                FL_DEBUG_ASSERT( p_neuron );

                // Get the error term $\frac{\partial E}{\partial \mathrm{net}_j}$
                const ValueT dEdNetj = sensitivies_.at(p_neuron);

                // Computes $\frac{\partial \mathrm{net}_j}{\partial w_{ij}}$
                NetInputFunction<ValueT>* p_netInFunc = p_neuron->getNetInputFunction();
                std::vector<ValueT> inputs = p_neuron->inputs();
                std::vector<ValueT> weights = p_neuron->weights();
				if (p_neuron->hasBias())
				{
					inputs.push_back(1);
					weights.push_back(p_neuron->getBias());
				}
                const std::vector<ValueT> dNetjdWs = p_netInFunc->evalDerivativeWrtWeight(inputs.begin(), inputs.end(), weights.begin(), weights.end());

                // For each incoming connection $i$, compute its error gradient term \frac{\partial E}{\partial w_{ij}}
				std::size_t i = 0;
				std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
                for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																		 connEndIt = inConns.end();
                     connIt != connEndIt;
                     ++connIt)
                {
                    Connection<ValueT>* p_conn = *connIt;

                    FL_DEBUG_ASSERT( p_conn );

                    const ValueT dNetjdWij = dNetjdWs[i];

					dEdWs_[p_conn] = dEdNetj*dNetjdWij;

					++i;
                }
				if (p_neuron->hasBias())
				{
					// Computes $\Delta w_{ij} = -\eta\frac{\partial E}{\partial w_{ij}}$
					const ValueT dNetjdWij = dNetjdWs[i];

					dEdBs_[p_neuron] = dEdNetj*dNetjdWij;

				}
            }
        }
    }

	private: void doResetSingleEpoch()
	{
		sensitivies_.clear();
		oldDeltaWs_.clear();
		oldDeltaBs_.clear();
		dEdWs_.clear();
		dEdBs_.clear();
	}


	private: std::map<const Neuron<ValueT>*,ValueT> sensitivies_;
	private: std::map<const Connection<ValueT>*,ValueT> oldDeltaWs_; // Only for momentum learning
	private: std::map<const Neuron<ValueT>*,ValueT> oldDeltaBs_; // Only for momentum learning
	private: std::map<const Connection<ValueT>*,ValueT> dEdWs_; // Only for offline training
	private: std::map<const Neuron<ValueT>*,ValueT> dEdBs_; // Only for offline training
}; // Jang1993HybridLearningAlgorithm
#endif

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_H
