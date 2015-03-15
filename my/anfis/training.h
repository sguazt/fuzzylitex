#ifndef FL_ANFIS_TRAINING_H
#define FL_ANFIS_TRAINING_H


#include <boost/noncopyable.hpp>
#include <cstddef>
#include <fl/fuzzylite.h>
#include <my/dataset.h>
#include <my/detail/traits.h>
#include <my/detail/iterators.h>
#include <my/ann/error_functions.h>
#include <my/ann/networks.h>
#include <vector>


namespace fl { namespace anfis {

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
class Jang1993HybridLearningAlgorithm: public fl::ann::TrainingAlgorithm<fl::scalar>
{
    public: Jang1993HybridLearningAlgorithm()
    {
    }

    /// Training for a single epoch
    private: void doTrainSingleEpoch(const DataSet<ValueT>& data)
    {
		Network<ValueT>* p_nnet = this->getNetwork();

		// check: null
		FL_DEBUG_ASSERT( p_nnet );

		ErrorFunction<ValueT>* p_errFunc = this->getErrorFunction();

		// check: null
		FL_DEBUG_ASSERT( p_errFunc );

        for (typename DataSet<ValueT>::ConstEntryIterator dseIt = data.entryBegin(),
                                                          dseEndIt = data.entryEnd();
             dseIt != dseEndIt;
             ++dseIt)
        {
            const DataSetEntry<ValueT>& entry = *dseIt;

////[XXX]
#ifdef FL_DEBUG
std::cerr << "Input: [";
std::copy(entry.inputBegin(), entry.inputEnd(), std::ostream_iterator<ValueT>(std::cerr, ", "));
std::cerr << "]," << std::endl;
std::cerr << "Target Output: [";
std::copy(entry.outputBegin(), entry.outputEnd(), std::ostream_iterator<ValueT>(std::cerr, ", "));
std::cerr << "]" << std::endl;
#endif // FL_DEBUG
////[/XXX]
            const std::size_t nout = entry.numOfOutputs();

            if (nout != p_nnet->numOfOutputs())
            {
                FL_THROW2(std::invalid_argument, "Incorrect output dimension");
            }

            const std::vector<ValueT> targetOut(entry.outputBegin(), entry.outputEnd());

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

}} // Namespace fl::anfis

#endif // FL_ANFIS_TRAINING_H
