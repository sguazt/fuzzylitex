#ifndef FL_ANN_LAYERS_H
#define FL_ANN_LAYERS_H


#include <cstddef>
#include <fl/fuzzylite.h>
#include <my/commons.h>
#include <my/ann/neurons.h>
#include <stdexcept>
#include <string>
#include <vector>


namespace fl { namespace ann {

template <typename T> class Network;

/**
 * Layer of neurons in a neural network.
 *
 * The Layer is essentially a container of neurons and it provides methods for manipulating neurons.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class Layer
{
	private: typedef std::vector<Neuron<ValueT>*> NeuronContainer;

	FL_MAKE_ITERATORS(public, NeuronContainer, Neuron, neuron, neurons_)


	/// Default constructor
	public: explicit Layer(Network<ValueT>* p_network = fl::null)
	: p_network_(p_network),
	  hasBias_(false)/*,
	  p_biasNeuron_(fl::null)*/
	{
	}

	/// Creats a layer containing the given collection of (pointers to) neurons
	public: template <typename IterT>
			Layer(IterT first, IterT last, Network<ValueT>* p_network = fl::null)
	: neurons_(first, last),
	  p_network_(p_network),
	  hasBias_(false)/*,
	  p_biasNeuron_(fl::null)*/
	{
	}

	/// Destruct this layer
	public: virtual ~Layer()
	{
		this->clear();
	}

	/// Clears the entire layer
	public: void clear()
	{
		//this->reset();

		this->clearNeurons();

		hasBias_ = false;
	}

	/// Resets the activation and input levels for all the neurons in this layer
	public: void reset()
	{
		for (std::size_t i = 0,
						 n = neurons_.size();
			 i < n;
			 ++i)
		{
			neurons_[i]->reset();
		}
	}

	/// Adds the given (pointer to a) neuron to this layer
	///FIXME: use unique_ptr
	public: void addNeuron(Neuron<ValueT>* p_neuron)
	{
		if (!p_neuron)
		{
			FL_THROW2(std::invalid_argument, "Neuron cannot be null!");
		}

		p_neuron->setLayer(this);
		p_neuron->setHasBias(hasBias_);

		neurons_.push_back(p_neuron);
	}

	public: NeuronIterator eraseNeuron(NeuronIterator it)
	{
		Neuron<ValueT>* p_neuron = *it;

		// check: null
		FL_DEBUG_ASSERT( p_neuron );

		delete p_neuron;

		return neurons_.erase(it);
	}

	public: NeuronIterator eraseNeuron(ConstNeuronIterator it)
	{
		Neuron<ValueT>* p_neuron = *it;

		// check: null
		FL_DEBUG_ASSERT( p_neuron );

		delete p_neuron;

		return neurons_.erase(it);
	}

	/// Gets the neuron at the given position \a pos
	public: Neuron<ValueT>* getNeuron(std::size_t pos) const
	{
		if (pos >= neurons_.size())
		{
			FL_THROW2(std::out_of_range, "Neuron position is out-of-range");
		}

		return neurons_[pos];
	}

	/// Removes all neurons inside this layer
	public: void clearNeurons()
	{
//		for (NeuronIterator neuronIt = neurons_.begin(),
//							neuronEndIt = neurons_.end();
//			 neuronIt != neuronEndIt;
//			 /*empty*/)
//		{
//			neuronIt = this->eraseNeuron(neuronIt);
//		}
//		neurons_.clear();
		const std::size_t n = neurons_.size();
		for (std::size_t i = 0; i < n; ++i)
		{
			if (neurons_[i])
			{
				delete neurons_[i];
			}
		}
		neurons_.clear();

	}

	/// Returns the number of neurons inside this layer
	public: std::size_t numOfNeurons() const
	{
		return neurons_.size();
	}

	/// Sets (a pointer to) the neural network containing this layer
	public: void setNetwork(Network<ValueT>* p_network)
	{
		p_network_ = p_network;
	}

	/// Returns (a pointer to) the neural network containing this layer
	public: Network<ValueT>* getNetwork() const
	{
		return p_network_;
	}

	public: void setHasBias(bool v)
	{
		if (hasBias_ ^ v)
		{
			hasBias_ = v;
			this->handleBias();
		}
	}

	public: bool hasBias() const
	{
		return hasBias_;
	}

	/// Performs calculaton for all the neurons inside this layer
	public: std::vector<ValueT> process()
	{
		this->handleBias();

		std::vector<ValueT> outs;

		for (NeuronIterator neuronIt = neurons_.begin(),
							neuronEndIt = neurons_.end();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			const ValueT out = p_neuron->process();

			outs.push_back(out);
		}

		return outs;
	}

	/// Tells if this layer has no neurons 
	public: bool isEmpty()
	{
		return neurons_.empty();
	}

	/// Update bias connections
	private: void handleBias()
	{
		for (NeuronIterator it = neurons_.begin(),
							end = neurons_.end();
			 it != end;
			 ++it)
		{
			Neuron<ValueT>* p_neuron = *it;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			p_neuron->setHasBias(hasBias_);
		}
	}


	private: std::vector<Neuron<ValueT>*> neurons_; ///< The collection of neurons in this layer
    private: Network<ValueT>* p_network_; ///< Pointer to the neural network to which this layer belongs
	private: bool hasBias_; ///< Tells if this layer has a bias unit or not
}; // Layer

}} // Namespace fl::ann

#endif // FL_ANN_LAYERS_H
