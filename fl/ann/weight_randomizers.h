#ifndef FL_ANN_WEIGHT_RANDOMIZERS_H
#define FL_ANN_WEIGHT_RANDOMIZERS_H


#include <boost/noncopyable.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <my/ann/connection.h>
#include <my/ann/layers.h>
#include <my/ann/networks.h>
#include <my/ann/neurons.h>
#include <my/commons.h>
//#include <my/detail/math.h>
#include <my/detail/random.h>
#include <utility>


namespace fl { namespace ann {

namespace detail {

/// Randomizes the weights of the input connections to the given layer
template <typename ValueT, typename EngineT>
void RandomizeLayer(Layer<ValueT>* p_layer, ValueT from, ValueT to, EngineT& eng)
{
	for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
												neuronEndIt = p_layer->neuronEnd();
		 neuronIt != neuronEndIt;
		 ++neuronIt)
	{
		Neuron<ValueT>* p_neuron = *neuronIt;

		// check: null
		FL_DEBUG_ASSERT( p_neuron );

		std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
		for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																 connEndIt = inConns.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			Connection<ValueT>* p_conn = *connIt;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			const ValueT w = fl::detail::RandUnif(from, to, eng);

			p_conn->setWeight(w);
		}
		if (p_neuron->hasBias())
		{
			const ValueT b = fl::detail::RandUnif(from, to, eng);

			p_neuron->setBias(b);
		}
	}
}

/// Computes the common output range of the activation functions of the given layer
template <typename ValueT>
std::pair<ValueT,ValueT> ComputeLayerOutputRange(Layer<ValueT>* p_layer)
{
	std::pair<ValueT,ValueT> bestRange = std::make_pair(std::numeric_limits<ValueT>::min(), std::numeric_limits<ValueT>::max());

	for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
												neuronEndIt = p_layer->neuronEnd();
		 neuronIt != neuronEndIt;
		 ++neuronIt)
	{
		Neuron<ValueT>* p_neuron = *neuronIt;

		// check: null
		FL_DEBUG_ASSERT( p_neuron );

		std::pair<ValueT,ValueT> range = p_neuron->getActivationFunction()->getOutputRange();
		if (range.first > bestRange.first)
		{
			bestRange.first = range.first;
		}
		if (range.second < bestRange.second)
		{
			bestRange.second = range.second;
		}
	}

//	if (detail::FloatTraits<ValueT>::EssentiallyEqual(bestRange.first, bestRange.second))
//	{
//		bestRange.first -= 1;
//		bestRange.second += 1;
//	}

	return bestRange;
}

} // Namespace detail

/**
 * Constant Randomization
 *
 * Initialize the weights to a fixed value.
 *
 * Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class ConstWeightRandomizer
{
	public: explicit ConstWeightRandomizer(ValueT value)
	: val_(value)
	{
	}

	public: template <typename EngineT>
			void randomize(Network<ValueT>& nnet, EngineT& eng)
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( eng );

		if (nnet.numOfLayers() < 3)
		{
			// Initialization is application to a network with at least 1 input layer, 1 hidden layer and 1 output layer
			return;
		}

		for (typename Network<ValueT>::LayerIterator layerIt = nnet.hiddenLayerBegin(),
													 layerEndIt = nnet.hiddenLayerEnd();
			 layerIt != layerEndIt;
			 ++layerIt)
		{
			Layer<ValueT>* p_layer = *layerIt;

			// check: null
			FL_DEBUG_ASSERT( p_layer );

			this->setWeights(p_layer);
		}
		this->setWeights(nnet.getOutputLayer());
	}

	private: void setWeights(Layer<ValueT>* p_layer)
	{
		// check: null
		FL_DEBUG_ASSERT( p_layer );

		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
													neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
			for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																	 connEndIt = inConns.end();
				 connIt != connEndIt;
				 ++connIt)
			{
				Connection<ValueT>* p_conn = *connIt;

				// check: null
				FL_DEBUG_ASSERT( p_conn );

				p_conn->setWeight(val_);
			}
			if (p_neuron->hasBias())
			{
				p_neuron->setBias(val_);
			}
		}
	}


	private: ValueT val_;
}; // ConstWeightRandomizer

/**
 * Hard Range Randomization
 *
 * Randomly initializes the weights of the given neural network with values in
 * the range of [\a from, \a to)
 *
 * Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class RangeWeightRandomizer
{
	public: explicit RangeWeightRandomizer(ValueT from = -0.05, ValueT to = 0.05)
	: from_(from),
	  to_(to)
	{
	}

	public: template <typename EngineT>
			void randomize(Network<ValueT>& nnet, EngineT& eng)
	{
		if (nnet.numOfLayers() < 3)
		{
			// Initialization is application to a network with at least 1 input layer, 1 hidden layer and 1 output layer
			return;
		}

		for (typename Network<ValueT>::LayerIterator layerIt = nnet.hiddenLayerBegin(),
													 layerEndIt = nnet.hiddenLayerEnd();
			 layerIt != layerEndIt;
			 ++layerIt)
		{
			Layer<ValueT>* p_layer = *layerIt;

			// check: null
			FL_DEBUG_ASSERT( p_layer );

			detail::RandomizeLayer(p_layer, from_, to_, eng);
		}

		detail::RandomizeLayer(nnet.getOutputLayer(), from_, to_, eng);
	}


	private: ValueT from_;
	private: ValueT to_;
}; // RangeWeightRandomizer

/**
 * Initialize the weights according to the Nguyen-Widrow's method.
 *
 * In the original Nguyen-Widrow work [1], the authors make the following
 * assumptions:
 * - they consider a two-layer neural network (i.e., 1 input layer, 1 hidden
 *   layer and 1 output layer),
 * - the activation function is the sigmoid, and
 * - the input space range from -1 to +1 in values.
 * .
 * Currently, there is no accepted way to generalize the method to networks
 * with any number of layers.
 *
 * To cope with the general case, we adapted the method similarly to the
 * Ecog framework (https://github.com/encog); specifically:
 * - First initialize the neural network with random weight values in a specific
 *   range.
 * - Next, calculate a value beta, as follows:
 *     \f[\beta = 0.7h^\frac{1}{i}\f]
 *   The variable \f$h\f$ represents the number of hidden neurons, whereas the
 *   variable \f$i\f$ represents the number of input neurons.
 * - Next, calculate the Euclidean norm for all of the weights on a layer.
 *   This is calculated as follows: 
 *     \f[n=\sqrt{\sum_{i=0}^{i<w_{max}}w_i^2}\f]
 * - Once the beta and norm values have been calculate, the weights are adjusted
 *   using the previously calculated:
 *     \f[w_i = \frac{\beta w_i}{n}\f]
 * - The same equation is used to adjust the thresholds (biases).
 * .
 *
 * References:
 * -# Nguyen, D., and B. Widrow, "Improving the learning speed of 2-layer neural networks by choosing initial values of the adaptive weights," Proceedings of the International Joint Conference on Neural Networks, Vol. 3, 1990, pp. 21-26.
 * -# S.N. Sivanandam and S.N. Deepa, "Introduction to Neural Networks using MATLAB 6.0," Tata McGraw-Hill Education, 2006.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class NguyenWidrowWeightRandomizer
{
	public: template <typename EngineT>
			void randomize(Network<ValueT>& nnet, EngineT& eng)
	{
		if (nnet.numOfLayers() < 3)
		{
			// Initialization is application to a network with at least 1 input layer, 1 hidden layer and 1 output layer
			return;
		}

		Layer<ValueT>* p_prevLayer = nnet.getInputLayer();
		for (typename Network<ValueT>::LayerIterator layerIt = nnet.hiddenLayerBegin(),
													 layerEndIt = nnet.hiddenLayerEnd();
			 layerIt != layerEndIt;
			 ++layerIt)
		{
			Layer<ValueT>* p_layer = *layerIt;

			// check: null
			FL_DEBUG_ASSERT( p_layer );

			Randomize(p_prevLayer, p_layer, eng);

			p_prevLayer = p_layer;
		}
		Randomize(p_prevLayer, nnet.getOutputLayer(), eng);
	}

	// This is the same procedure as described in [2] (sec. 8.2.4, pag. 190)
	private: template <typename EngineT>
			 static void Randomize(Layer<ValueT>* p_prevLayer, Layer<ValueT>* p_layer, EngineT& eng)
	{
		const std::size_t numHiddenNeurons = p_layer->numOfNeurons(); // Num of neurons of this layer (without biases)
		const std::size_t numInputNeurons = p_prevLayer->numOfNeurons(); // Num of neurons of previous layer (without biases)

		const ValueT beta = 0.7*std::pow(numHiddenNeurons, (1.0/numInputNeurons));

		// Randomize weights in [-0.5,0.5] and biases in [-beta,beta]
		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
													neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
			for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																	 connEndIt = inConns.end();
				 connIt != connEndIt;
				 ++connIt)
			{
				Connection<ValueT>* p_conn = *connIt;

				// check: null
				FL_DEBUG_ASSERT( p_conn );

				const ValueT w = fl::detail::RandUnif(-0.5, 0.5, eng);

				p_conn->setWeight(w);
			}
		}

		// Compute weights' norm
		const ValueT norm = ComputeLayerWeightsNorm(p_layer);

		// Compute scaling factor
		const ValueT scale = beta/norm;

		// Reinitialize weights and bias
		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
													neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
			for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																	 connEndIt = inConns.end();
				 connIt != connEndIt;
				 ++connIt)
			{
				Connection<ValueT>* p_conn = *connIt;

				// check: null
				FL_DEBUG_ASSERT( p_conn );

				const ValueT w = scale*p_conn->getWeight();

				p_conn->setWeight(w);
			}

			if (p_neuron->hasBias())
			{
				const ValueT wb = fl::detail::RandUnif(-beta, beta, eng);

				p_neuron->setBias(wb);
			}
		}

	}

/* The following is an attempt to mimic the MATLAB's initnw function... still in progress
	private: template <typename EngineT>
			 static void Randomize(Layer<ValueT>* p_prevLayer, Layer<ValueT>* p_layer, EngineT& eng)
	{
		//const std::pair<ValueT,ValueT> range = detail::ComputeLayerOutputRange<ValueT>(p_layer);
		const std::size_t numHiddenNeurons = p_layer->numOfNeurons(); // Num of neurons of this layer (without biases)
		const std::size_t numInputNeurons = p_prevLayer->numOfNeurons(); // Num of neurons of previous layer (without biases)

		//const ValueT beta = 0.7*std::pow(numHiddenNeurons, (1.0/numInputNeurons))/(range.second-range.first);
		const ValueT beta = 0.7*std::pow(numHiddenNeurons, (1.0/numInputNeurons));

		//detail::RandomizeLayer(p_layer, -beta, beta, eng);

		// The original Nguyen-Widrow method assumes that inputs are in [-1, 1] interval
		const std::vector<ValueT> linspace = detail::LinSpace(-1, +1, p_layer->numOfNeurons());
		std::size_t neuronIdx = 0;
		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
													neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			const std::pair<ValueT,ValueT> activeInputRange = p_neuron->getActivationFunction()->getActiveInputRange();
			const ValueT netInScale = 0.5*(activeInputRange.second-activeInputRange.first);
			const ValueT netInOffs = 0.5*(activeInputRange.second+activeInputRange.first);
			const std::pair<ValueT,ValueT> outputRange = p_neuron->getActivationFunction()->getOutputRange();
			const ValueT inScale = 2.0/(outputRange.second-outputRange.first);
			const ValueT inOffs = 1-outputRange.second*inScale;
			short sign = 0;
			std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
			for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																	 connEndIt = inConns.end();
				 connIt != connEndIt;
				 ++connIt)
			{
				Connection<ValueT>* p_conn = *connIt;

				// check: null
				FL_DEBUG_ASSERT( p_conn );

				ValueT w = beta*fl::detail::RandUnif(-1, +1, eng);

				if (w > 0)
				{
					sign = 1;
				}
				else if (w < 0)
				{
					sign = -1; 
				}

				w *= netInScale; // Conversion of net inputs of [-1,1] to active input range
				w *= inScale; // Conversion of inputs of output range to [-1 1]

				p_conn->setWeight(w);
			}

			ValueT wb = beta*linspace[neuronIdx]*static_cast<ValueT>(sign);
			wb = netInScale*wb+netInOffs; // Conversion of net inputs of [-1,1] to active input range
			wb += w*inOffs; // Conversion of inputs of output range to [-1 1]

			p_neuron->setBias(wb);

			++neuronIdx;
		}

		const ValueT norm = ComputeLayerWeightsNorm(p_layer);

		detail::ScaleLayerWeights(p_layer, beta/norm);
	}

	private: template <typename EngineT>
			 static void RandomizeLayer(Layer<ValueT>* p_layer, ValueT from, ValueT to, EngineT& eng)
	{
	}

	/// Scales the weights of the input connections to the given layer
	static: void ScaleLayerWeights(Layer<ValueT>* p_layer, ValueT factor)
	{
		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
													neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
			for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																	 connEndIt = inConns.end();
				 connIt != connEndIt;
				 ++connIt)
			{
				Connection<ValueT>* p_conn = *connIt;

				// check: null
				FL_DEBUG_ASSERT( p_conn );

				p_conn->setWeight(factor*p_conn->getWeight());
			}
		}
	}
*/

	/// Computes the Euclidean norm of the weights of the input connections to the given layer
	private: static ValueT ComputeLayerWeightsNorm(Layer<ValueT>* p_layer)
	{
		ValueT norm = 0;
		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
													neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			const std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
			for (typename std::vector<Connection<ValueT>*>::const_iterator connIt = inConns.begin(),
																		   connEndIt = inConns.end();
				 connIt != connEndIt;
				 ++connIt)
			{
				const Connection<ValueT>* p_conn = *connIt;

				// check: null
				FL_DEBUG_ASSERT( p_conn );

				const ValueT w = p_conn->getWeight();

				norm += w*w;
			}
		}

		return std::sqrt(norm);
	}
}; // NguyenWidrowWeightRandomizer

/**
 * Gaussian Randomization
 *
 * Initialize the weights randomly according to a normal distribution with mean 0 and standard deviation \a sd
 *
 * Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class GaussianWeightRandomizer
{
	public: explicit GaussianWeightRandomizer(ValueT sd)
	: sd_(sd)
	{
	}

	public: template <typename EngineT>
			void randomize(Network<ValueT>& nnet, EngineT& eng)
	{
		FL_SUPPRESS_UNUSED_VARIABLE_WARNING( eng );

		if (nnet.numOfLayers() < 3)
		{
			// Initialization is application to a network with at least 1 input layer, 1 hidden layer and 1 output layer
			return;
		}

		for (typename Network<ValueT>::LayerIterator layerIt = nnet.hiddenLayerBegin(),
													 layerEndIt = nnet.hiddenLayerEnd();
			 layerIt != layerEndIt;
			 ++layerIt)
		{
			Layer<ValueT>* p_layer = *layerIt;

			// check: null
			FL_DEBUG_ASSERT( p_layer );

			this->setWeights(p_layer, eng);
		}
		this->setWeights(nnet.getOutputLayer(), eng);
	}

	private: template <typename EngineT>
			 void setWeights(Layer<ValueT>* p_layer, EngineT& eng)
	{
		// check: null
		FL_DEBUG_ASSERT( p_layer );

		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_layer->neuronBegin(),
													neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			std::vector<Connection<ValueT>*> inConns = p_neuron->inputConnections();
			for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																	 connEndIt = inConns.end();
				 connIt != connEndIt;
				 ++connIt)
			{
				Connection<ValueT>* p_conn = *connIt;

				// check: null
				FL_DEBUG_ASSERT( p_conn );

				const ValueT w = fl::detail::RandNormal(0.0, sd_, eng);

				p_conn->setWeight(w);
			}
			if (p_neuron->hasBias())
			{
				const ValueT b = fl::detail::RandNormal(0.0, sd_, eng);

				p_neuron->setBias(b);
			}
		}
	}


	private: ValueT sd_;
}; // GaussianWeightRandomizer

}} // Namespace fl::ann

#endif // FL_ANN_WEIGHT_RANDOMIZERS_H
