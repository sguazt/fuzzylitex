#ifndef FL_ANN_TRAINING_ALGORITHMS_H
#define FL_ANN_TRAINING_ALGORITHMS_H


#include <boost/noncopyable.hpp>
#include <cstddef>
#include <fl/fuzzylite.h>
#include <my/dataset.h>
#include <my/detail/float_traits.h>
#include <my/detail/iterators.h>
#include <my/ann/error_functions.h>
#include <my/ann/networks.h>
#include <vector>


namespace fl { namespace ann {

/**
 * Base class for training algorithms
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class TrainingAlgorithm: boost::noncopyable
{
    protected: typedef FL_ForwardIteratorType(ValueT) FwdIter;


    public: TrainingAlgorithm()
    : p_nnet_(fl::null),
      p_errFunc_(new SumSquaredErrorFunction<ValueT>()),
      maxErr_(1e-3)
    {
    }

    public: virtual ~TrainingAlgorithm() { }

    public: void setNetwork(Network<ValueT>* p_nnet)
    {
        p_nnet_ = p_nnet;
    }

    public: Network<ValueT>* getNetwork() const
    {
        return p_nnet_;
    }

    public: void setErrorFunction(ErrorFunction<ValueT>* p_errFunc)
    {
		p_errFunc_.reset(p_errFunc);
    }

    public: ErrorFunction<ValueT>* getErrorFunction() const
    {
        return p_errFunc_.get();
    }

    public: void setMaxError(ValueT v)
    {
        maxErr_ = v;
    }

    public: ValueT getMaxError() const
    {
        return maxErr_;
    }

//  public: void setMaxNumOfIterations(ValueT v)
//  {
//      maxIt_ = v;
//  }

//  public: ValueT getMaxNumOfIterations() const
//  {
//      return maxIt_;
//  }

    public: void train(const DataSet<ValueT>& data)
    {
//        while (...)
//        {
//            this->trainSingleEpoch(data);
//        }
    }

    public: ValueT trainSingleEpoch(const DataSet<ValueT>& data)
    {
        if (!p_nnet_)
        {
            FL_THROW("Cannot train a null neural network");
        }
        if (!p_errFunc_.get())
        {
            FL_THROW("Cannot train with a null error function");
        }

		p_errFunc_->reset();

        this->doTrainSingleEpoch(data);

		return p_errFunc_->getTotalError();
    }

    private: virtual void doTrainSingleEpoch(const DataSet<ValueT>& data) = 0;


    private: Network<ValueT>* p_nnet_;
    private: FL_unique_ptr< ErrorFunction<ValueT> > p_errFunc_;
    private: ValueT maxErr_; ///< Stop condition based on max allowed network error
    //private: std::size_t maxIt_; ///< Stop condition based on max allowed number of iterations
	private: ValueT totalErr_;
}; // TrainingAlgorithm


/**
 * The stochastic gradient descent backpropagation learning algorithm with momentum
 * 
 * In the backpropagation learning algorithm, there are two phases in its
 * learning cycle:
 * 1. the feedforward phase, where the input pattern is propagated through the
 *    network, and
 * 2. the backward phase, where the error between the desired and obtained
 *    output is back-propagated through the network in order to change the
 *    weights and then to decrease the error.
 *
 * The stochastic gradient descent backpropagation algorithm (also known as
 * incremental gradient descent algorithm) differs from the traditional gradient
 * descent backpropagation algorithm in the way weights are updated.
 * Where as the gradient descent training rule computes weight updates after
 * summing over all the training examples, the idea behind the stochastic
 * gradient descent is to approximate this gradient descent search by updating
 * weights incrementally, following the calculation of the error for each
 * individual example [1].
 * One of the advantage of the stochastic gradient descent over the standard
 * gradient descent descent is that the stochastic version can sometimes avoid
 * falling into local minima of the error function.
 *
 * Intuitively, the stochastic gradient descent backpropagation algorithm can be
 * outlined as follows [1]:
 *   Given a set of input patterns (the training set)
 *   WHILE the neural network can consecutively map all patterns correctly DO
 *    FOR EACH input pattern x, repeat the following Steps 1 to 3 until the output vector y is equal (or close enough) to the target vector t for the given input vector x.
 *     1. Input x to the neural network.
 *     2. (Feedforward)
 *        Go through the neural network, from the input to hidden layers, then from the hidden to output layers, and get output vector y.
 *     3. (Backward propagation of error corrections)
 *        IF y is equal or close enough to t, THEN
 *         BREAK // Go back to the beginning of the outer WHILE loop.
 *        ELSE
 *         Backpropagate through the neural network and adjust the weights so that the next y is closer to t, that is:
 *         \f[
 *          w_{ij}(n) = w_{ij}(n-1) + \Delta w_{ij}(n-1)
 *         \f]
 *         where \f$w_{ij}(n)\f$ is the weight on the connection from neuron \f$i\f$ to neuron \f$j\f$ at step \f$n\f$.
 *        END IF
 *   END WHILE
 * In the above, each WHILE loop iteration is called an <em>epoch</em>.
 * An epoch is one cycle through the entire set of patterns under consideration.
 * Each variant of the backpropagation algorithm differs from the way
 * \f$\Delta w_{ij}(n)\f$ is defined.
 *
 * In the <em>gradient-descent backpropagation learning algorithm</em>, the
 * weights \f$w_{ij}(n)$  are updated in a way that is proportional to the gradient
 * of the error function, that is:
 * \f[
 *  \Delta w_{ij}(n) = -\eta \frac{\partial E}{\partial w_{ij}} + \alpha \Delta w_{ij}(n-1)
 * \f]
 * where:
 * - \f$E\f$ is the <em>error function</em>, \f$\partial\f$ is the partial
 *   derivative operator,
 * - \f$\eta\f$ is a positive constant called <em>learning rate</em> (whose
 *   purpose is to limit the degree to which weights are changed at each step),
 * - \f$\alpha\f$ is a constant in [0,1) called <em>momentum</em> (whose purpose
 *   is to keep gradient descent search trajectory in the same direction from
 *   one iteration to the next).
 * .
 *
 * Theoretically, the use of the momemtum term should provide the search process
 * with a kind of inertia and could help to avoid excessive oscillations in
 * narrow valleys of the error function [2].
 *
 * The use of the momentum can sometimes have the effect of keeping the gradient
 * descent search trajectory through small local minima in the error surface, or
 * along flat regions in the surface where the gradient descent search
 * trajectory would stop if there were no momentum [1].
 * It also has the effect of gradually increasing the step size of the search in
 * regions where the gradient is unchanging, thereby speeding convergence [1].
 *
 * References:
 * -# T.M. Mitchell, "Machine Learning," McGraw-Hill, 1997.
 * -# R. Rojas, "Neural Networks: A Sistematic Introduction," Springer, 1996.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class GradientDescentBackpropagationAlgorithm: public TrainingAlgorithm<ValueT>
{
	//FIXME:  Move error terms inside each neuron into this class

    public: GradientDescentBackpropagationAlgorithm()
    : learnRate_(0.01),
	  momentum_(0.9),
	  maxEpochs_(1)
    {
    }

    public: void setLearningRate(ValueT v)
    {
        learnRate_ = detail::FloatTraits<ValueT>::DefinitelyMax(v, 0);
    }

    public: ValueT getLearningRate() const
    {
        return learnRate_;
    }

    public: void setMomentum(ValueT v)
    {
        momentum_ = detail::FloatTraits<ValueT>::Clamp(v, 0, 1);
    }

    public: ValueT getMomentum() const
    {
        return momentum_;
    }

    public: void setMaxEpochs(std::size_t n)
    {
        maxEpochs_ = n;
    }

    public: std::size_t getMaxEpochs() const
    {
        return maxEpochs_;
    }

    /// Training for a single epoch
    private: void doTrainSingleEpoch(const DataSet<ValueT>& data)
    {
        Network<ValueT>* p_nnet = this->getNetwork();
        ErrorFunction<ValueT>* p_errFunc = this->getErrorFunction();

		// check: null
		FL_DEBUG_ASSERT( p_nnet );
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

//FL_DEBUG_TRACE("Update weight step");//XXX
            // 2.3. Update network weights
            this->updateWeights(targetOut, actualOut);

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

                // Updates the error term of this neuron
                p_neuron->setError(dEdOj*dOjdNetj);
//////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "Output neuron (" << p_neuron << ") - dEdOj: " << dEdOj << " - dOjdNetj: " << dOjdNetj << " -> Error: " << p_neuron->getError() << std::endl;
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
                        const ValueT dEdNetk = p_toNeuron->getError();

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

                    p_neuron->setError(dEdNetj);
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "Hidden neuron (" << p_neuron << ") - dOjdNetj: " << dOjdNetj << " - dEdNetkdNetkdOjs: " << dEdNetkdNetkdOjs << " -> Error: " << p_neuron->getError() << std::endl;
//#endif // FL_DEBUG
////[/XXX]

					++j;
                }
            }
        }
    }

    private: void updateWeights(const std::vector<ValueT>& targetOut, const std::vector<ValueT>& actualOut)
    {
		Network<ValueT>* p_nnet = this->getNetwork();
		ErrorFunction<ValueT>* p_errFunc = this->getErrorFunction();
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
                    const ValueT dEdNetj = p_neuron->getError();

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
                const ValueT dEdNetj = p_neuron->getError();

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


    private: ValueT learnRate_ ; ///< The learning rate parameter
    private: ValueT momentum_ ; ///< The momentum_ parameter
    private: std::size_t maxEpochs_; ///< Maximum number of epochs to train
	private: std::map<const Connection<ValueT>*,ValueT> oldDeltaWs_;
	private: std::map<const Neuron<ValueT>*,ValueT> oldDeltaBs_;
}; // GradientDescentBackprogrationAlgorithm

}} // Namespace fl::ann

#endif // FL_ANN_TRAINING_ALGORITHMS_H
