/**
 * \file fl/ann/netowrks.h
 *
 * \brief Artificial neural networks
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

#ifndef FL_ANN_NETWORKS_H
#define FL_ANN_NETWORKS_H


#include <cstddef>
#include <cstdlib>
#include <fl/ann/connection.h>
#include <fl/ann/layers.h>
#include <fl/ann/neurons.h>
#include <fl/commons.h>
#include <fl/detail/random.h>
#include <fl/fuzzylite.h>
#include <map>
#include <string>
#include <utility>
#include <vector>


namespace fl { namespace ann {

/**
 * Base class for artificial neural networks.
 *
 * It provides generic structure and functionality for different types of neural
 * networks.
 * A neural network can be characterized by a collection of neurons (organized
 * in layers) and by a learning rule.
 * Custom neural networks are created by deriving from this class, creating
 * layers of interconnected network specific neurons, and setting network
 * specific learning rule.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class Network
{
    private: typedef std::pair<const Neuron<ValueT>*,const Neuron<ValueT>*> ConnectionId;
    private: typedef std::map<ConnectionId,Connection<ValueT>*> ConnectionMap;
	private: typedef typename ConnectionMap::iterator ConnectionIterator;
	private: typedef typename ConnectionMap::const_iterator ConstConnectionIterator;
	private: typedef std::vector<Layer<ValueT>*> LayerContainer;
	public: typedef ValueT ValueType;

	FL_MAKE_ITERATORS(public, LayerContainer, Layer, hiddenLayer, hiddenLayers_)


	/// Creates an empty neural network.
	public: Network()
	/*: p_inLayer_(fl::null),
	  p_outLayer_(fl::null)*/
	{
	}

	/// Destroy this network
	public: virtual ~Network()
	{
		this->clear();
	}

	/// Clears the entire network
	public: void clear()
	{
		p_inLayer_.reset();

		for (LayerIterator layerIt = hiddenLayers_.begin(),
						   layerEndIt = hiddenLayers_.end();
			 layerIt != layerEndIt;
			 ++layerIt)
		{
			Layer<ValueT>* p_layer = *layerIt;

			//check: null
			FL_DEBUG_ASSERT( p_layer );

			delete p_layer;
		}
		hiddenLayers_.clear();

		p_outLayer_.reset();

		for (typename ConnectionMap::iterator connIt = conns_.begin(),
											  connEndIt = conns_.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			Connection<ValueT>* p_conn = connIt->second;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			delete p_conn;
		}
		conns_.clear();
	}

	/**
 	 * Resets the activation and input levels for all the neurons in the layers
	 * in the entire network.
	 */
	public: void reset()
	{
		p_inLayer_->reset();

		for (LayerIterator layerIt = hiddenLayers_.begin(),
						   layerEndIt = hiddenLayers_.end();
			 layerIt != layerEndIt;
			 ++layerIt)
		{
			Layer<ValueT>* p_layer = *layerIt;

			//check: null
			FL_DEBUG_ASSERT( p_layer );

			p_layer->reset();
		}

		p_outLayer_->reset();

		for (typename ConnectionMap::iterator connIt = conns_.begin(),
											  connEndIt = conns_.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			Connection<ValueT>* p_conn = connIt->second;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			p_conn->reset();
		}
	}

	/// Adds the given (pointer to a) layer to this network
	///FIXME: use unique_ptr
	public: void addHiddenLayer(Layer<ValueT>* p_layer)
	{
		if (!p_layer)
		{
			FL_THROW2(std::invalid_argument, "Hidden layer cannot be null");
		}

		// set parent network for added layer
		p_layer->setNetwork(this);

		// add layer to layers collection
		hiddenLayers_.push_back(p_layer);
	}

	/**
	 * Returns the number of hidden layers in this network
	 *
	 * \return the number of layers
	 */
	public: std::size_t numOfHiddenLayers() const
	{
		return hiddenLayers_.size();
	}

	/**
	 * Sets the input layer
	 *
	 * \param p_layer the input layer
	 */
	public: void setInputLayer(Layer<ValueT>* p_layer)
	{
		p_inLayer_.reset(p_layer);
	}

	/**
	 * Returns the input neurons
	 *
	 * \return the input neurons
	 */
	public: Layer<ValueT>* getInputLayer() const
	{
		return p_inLayer_.get();
	}

	/**
	 * Gets the number of input neurons
	 *
	 * \return number of input neurons
	 */
	public: std::size_t numOfInputs() const
	{
		return p_inLayer_->numOfNeurons();
	}

	/**
	 * Sets the output layer
	 *
	 * \param p_layer the output layer
	 */
	public: void setOutputLayer(Layer<ValueT>* p_layer)
	{
		p_outLayer_.reset(p_layer);
	}

	/**
	 * Returns the output neurons
	 *
	 * \return the output neurons
	 */
	public: Layer<ValueT>* getOutputLayer() const
	{
		return p_outLayer_.get();
	}

	/**
	 * Gets the number of output neurons
	 *
	 * \return number of output neurons
	 */
	public: std::size_t numOfOutputs() const
	{
		return p_outLayer_->numOfNeurons();
	}

	/**
	 * Returns the number of all the layers in this network
	 *
	 * \return the number of layers
	 */
	public: std::size_t numOfLayers() const
	{
		return hiddenLayers_.size() + (p_inLayer_.get() ? 1 : 0) + (p_outLayer_.get() ? 1 : 0);
	}

//	/**
//	 * Sets network input. Input is an array of double values.
//	 *
//	 * @param inputVector network input as double array
//	 */
//	public: template <typename IterT>
//			void setInput(IterT first, IterT last)
//	{
//		input_.assign(first, last);
//
//		if (input_.length != inputNeurons_.size())
//		{
//			FL_THROW("Input vector size does not match network input dimension");
//		}
//
//		for (std::size_t i = 0,
//						 n = inputNeurons_.size();
//			 i < n;
//			 ++i)
//		{
//			Neuron* p_neuron = inputNeurons_[i];
//
//			// check: null
//			FL_DEBUG_ASSERT( p_neuron );
//
//			p_neuron->setInput(input_[i]);
//		}
//	}

	/**
	 * Gets the layer at the given index \a idx.
	 *
	 * The input layer has index 0, the first hidden layer has index 1, ...
	 * The index of the output layer is given by 1+<number of hidden layers>.
	 */
	public: Layer<ValueT>* getLayer(std::size_t idx) const
	{
		// pre: 0 <= idx <= total number of layers (input+hiddens+output)
		if (idx > (hiddenLayers_.size()+1))
		{
			FL_THROW2(std::out_of_range, "Layer index is out-of-range");
		}

		if (idx == 0)
		{
			return this->p_inLayer_.get();
		}
		if (idx == (hiddenLayers_.size()+1))
		{
			return this->p_outLayer_.get();
		}
		return hiddenLayers_.at(idx-1);
	}

	/// Set the network input
	public: template <typename IterT>
			void setInput(IterT first, IterT last)
	{
		const std::size_t n = std::distance(first, last);

		if (n != this->numOfInputs())
		{
			FL_THROW("Input vector size does not match network input dimension");
		}

		for (typename Layer<ValueT>::NeuronIterator neuronIt = p_inLayer_->neuronBegin(),
													neuronEndIt = p_inLayer_->neuronEnd();
			 neuronIt != neuronEndIt && first != last;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			p_neuron->setNetInput(*first++);
		}
	}

	/**
	 * Returns the current network input vector.
	 *
	 * \return the network input vector
	 */
	public: std::vector<ValueT> getInput() const
	{
		std::vector<ValueT> input;

		for (typename Layer<ValueT>::ConstNeuronIterator neuronIt = p_inLayer_->neuronBegin(),
														 neuronEndIt = p_inLayer_->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			input.push_back(p_neuron->getInput());
		}

		return input;
	}

	/// Connects \a p_from neuron with \a p_to neuron and assigns a weight of \a weight
	public: void connect(Neuron<ValueT>* p_from, Neuron<ValueT>* p_to, ValueT weight)
	{
		// pre: p_from != null && p_to != null
		if (!p_from)
		{
			FL_THROW2(std::invalid_argument, "The 'from' neuron cannot be null in a connection");
		}
		if (!p_to)
		{
			FL_THROW2(std::invalid_argument, "The 'to' neuron cannot be null in a connection");
		}
//FL_DEBUG_TRACE("Connection (" << p_from << ") -> (" << p_to << ")");//XXX

		ConnectionId connId = MakeConnectionId(p_from, p_to);

		if (conns_.count(connId) == 0)
		{
			Connection<ValueT>* p_conn = new Connection<ValueT>(p_from, p_to, weight);

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			conns_[connId] = p_conn;

			//p_from->addOutputConnection(p_conn);
			//p_to->addInputConnection(p_conn);
		}
		else
		{
			conns_[connId]->setWeight(weight);
		}
	}

	/// Fully connects \a p_from layer to \a p_to layer and assigns a weight of \a weight
	public: void connect(Layer<ValueT>* p_from, Layer<ValueT>* p_to, ValueT weight)
	{
		// pre: p_from != null && p_to != null
		if (!p_from)
		{
			FL_THROW2(std::invalid_argument, "The 'from' layer cannot be null in a connection");
		}
		if (!p_to)
		{
			FL_THROW2(std::invalid_argument, "The 'to' layer cannot be null in a connection");
		}

		for (typename Layer<ValueT>::NeuronIterator fromNeuronIt = p_from->neuronBegin(),
													fromNeuronEndIt = p_from->neuronEnd();
			 fromNeuronIt != fromNeuronEndIt;
			 ++fromNeuronIt)
		{
			Neuron<ValueT>* p_fromNeuron = *fromNeuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_fromNeuron );

			for (typename Layer<ValueT>::NeuronIterator toNeuronIt = p_to->neuronBegin(),
														toNeuronEndIt = p_to->neuronEnd();
				 toNeuronIt != toNeuronEndIt;
				 ++toNeuronIt)
			{
				Neuron<ValueT>* p_toNeuron = *toNeuronIt;

				// check: null
				FL_DEBUG_ASSERT( p_toNeuron );

				this->connect(p_fromNeuron, p_toNeuron, weight);
			}
		}
	}

	/// Removes the connection between \a p_from neuron and \a p_to neuron.
	public: void disconnect(Neuron<ValueT>* p_from, Neuron<ValueT>* p_to)
	{
		// pre: p_from != null && p_to != null
		if (!p_from)
		{
			FL_THROW2(std::invalid_argument, "The 'from' neuron cannot be null in a connection");
		}
		if (!p_to)
		{
			FL_THROW2(std::invalid_argument, "The 'to' neuron cannot be null in a connection");
		}

		ConnectionId connId = MakeConnectionId(p_from, p_to);

		if (conns_.count(connId) > 0)
		{
			Connection<ValueT>* p_conn = conns_.at(connId);

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			delete p_conn;

			conns_.erase(connId);
		}
	}

	/// Fully disconnects \a p_from layer from \a p_to layer
	public: void disconnect(Layer<ValueT>* p_from, Layer<ValueT>* p_to)
	{
		// pre: p_from != null && p_to != null
		if (!p_from)
		{
			FL_THROW2(std::invalid_argument, "The 'from' layer cannot be null in a connection");
		}
		if (!p_to)
		{
			FL_THROW2(std::invalid_argument, "The 'to' layer cannot be null in a connection");
		}

		for (typename Layer<ValueT>::NeuronIterator fromNeuronIt = p_from->neuronBegin(),
													fromNeuronEndIt = p_from->neuronEnd();
			 fromNeuronIt != fromNeuronEndIt;
			 ++fromNeuronIt)
		{
			Neuron<ValueT>* p_fromNeuron = *fromNeuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_fromNeuron );

			for (typename Layer<ValueT>::NeuronIterator toNeuronIt = p_to->neuronBegin(),
														toNeuronEndIt = p_to->neuronEnd();
				 toNeuronIt != toNeuronEndIt;
				 ++toNeuronIt)
			{
				Neuron<ValueT>* p_toNeuron = *toNeuronIt;

				// check: null
				FL_DEBUG_ASSERT( p_toNeuron );

				this->disconnect(p_fromNeuron, p_toNeuron);
			}
		}
	}

	/// Removes all the output connections from \a p_from neuron.
	public: void disconnectFrom(Neuron<ValueT>* p_from)
	{
		// pre: null
		if (!p_from)
		{
			FL_THROW2(std::invalid_argument, "The 'from' neuron cannot be null in a connection");
		}

		//FIXME: this is very inefficient. Find a better way to do it
		const std::vector<Connection<ValueT>*> outConns = this->outputConnections(p_from);
		for (typename std::vector<Connection<ValueT>*>::const_iterator connIt = outConns.begin(),
																	   connEndIt = outConns.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			const Connection<ValueT>* p_conn = *connIt;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			this->disconnect(p_from, p_conn->getToNeuron());
		}
	}

	/// Removes all the input connections to \a p_to neuron.
	public: void disconnectTo(Neuron<ValueT>* p_to)
	{
		// pre: null
		if (!p_to)
		{
			FL_THROW2(std::invalid_argument, "The 'to' neuron cannot be null in a connection");
		}

		//FIXME: this is very inefficient. Find a better way to do it
		const std::vector<Connection<ValueT>*> inConns = this->inputConnections(p_to);
		for (typename std::vector<Connection<ValueT>*>::const_iterator connIt = inConns.begin(),
																	   connEndIt = inConns.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			const Connection<ValueT>* p_conn = *connIt;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			this->disconnect(p_conn->getFromNeuron(), p_to);
		}
	}

	/// Returns the connection from \a p_from neuron to \a p_to neuron
	public: Connection<ValueT>* getConnection(const Neuron<ValueT>* p_from, const Neuron<ValueT>* p_to) const
	{
		if (this->isConnected(p_from, p_to))
		{
			return conns_.at(MakeConnectionId(p_from, p_to));
		}

		return fl::null;
	}

	/// Tells if there is a connection from neuron \a p_from to neuron \a p_to
	public: bool isConnected(const Neuron<ValueT>* p_from, const Neuron<ValueT>* p_to) const
	{
		// pre: p_from != null && p_to != null
		if (!p_from)
		{
			FL_THROW2(std::invalid_argument, "Cannot get the input connecction from a 'null' neuron'");
		}
		if (!p_to)
		{
			FL_THROW2(std::invalid_argument, "Cannot get the input connecction to a 'null' neuron'");
		}

		return conns_.count(MakeConnectionId(p_from, p_to)) > 0;
	}

	/// Returns all the input connections to the \a p_to neuron
	public: std::vector<Connection<ValueT>*> inputConnections(const Neuron<ValueT>* p_neuron)
	{
		// pre: p_neuron != null
		if (!p_neuron)
		{
			FL_THROW2(std::invalid_argument, "Cannot get the input connections to a 'null' neuron");
		}

		std::vector<Connection<ValueT>*> conns;

		//FIXME: this is very inefficient. Find a better way to do it
		for (ConnectionIterator connIt = conns_.begin(),
								connEndIt = conns_.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			Connection<ValueT>* p_conn = connIt->second;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			if (p_conn->getToNeuron() != p_neuron)
			{
				continue;
			}

			conns.push_back(p_conn);
		}

		return conns;
	}

	/// Returns all the output connections from the \a p_to neuron
	public: std::vector<Connection<ValueT>*> outputConnections(const Neuron<ValueT>* p_neuron)
	{
		// pre: p_neuron != null
		if (!p_neuron)
		{
			FL_THROW2(std::invalid_argument, "Cannot get the output connection to a 'null' neuron");
		}

		std::vector<Connection<ValueT>*> conns;

		//FIXME: this is very inefficient. Find a better way to do it
		for (ConnectionIterator connIt = conns_.begin(),
								connEndIt = conns_.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			Connection<ValueT>* p_conn = connIt->second;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			if (p_conn->getFromNeuron() != p_neuron)
			{
				continue;
			}

			conns.push_back(p_conn);
		}

		return conns;
	}

	/**
	 * Returns the network output vector.
	 *
	 * \return the network output vector
	 */
	public: std::vector<ValueT> getOutput() const
	{
		std::vector<ValueT> output;

		for (typename Layer<ValueT>::ConstNeuronIterator neuronIt = p_outLayer_->neuronBegin(),
														 neuronEndIt = p_outLayer_->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;

			// check: null
			FL_DEBUG_ASSERT( p_neuron );

			output.push_back(p_neuron->getOutput());
		}

		return output;
	}

	public: std::vector<ValueT> process()
	{
		//check: null
		FL_DEBUG_ASSERT( p_inLayer_.get() );

		p_inLayer_->process();

		for (LayerIterator layerIt = hiddenLayers_.begin(),
						   layerEndIt = hiddenLayers_.end();
			 layerIt != layerEndIt;
			 ++layerIt)
		{
			Layer<ValueT>* p_layer = *layerIt;

			//check: null
			FL_DEBUG_ASSERT( p_layer );

			p_layer->process();
		}

		//check: null
		FL_DEBUG_ASSERT( p_outLayer_.get() );

		return p_outLayer_->process();
	}

	/// Evaluates the given input pattern by forwarding it through the network and returning the related output
	public: template <typename IterT>
			std::vector<ValueT> process(IterT first, IterT last)
	{
		this->setInput(first, last);

		return this->process();
	}

	private: static std::pair<const Neuron<ValueT>*,const Neuron<ValueT>*> MakeConnectionId(Neuron<ValueT>* p_from, Neuron<ValueT>* p_to)
	{
		return std::make_pair(static_cast<const Neuron<ValueT>*>(p_from), static_cast<const Neuron<ValueT>*>(p_to));
	}

	private: static std::pair<const Neuron<ValueT>*,const Neuron<ValueT>*> MakeConnectionId(const Neuron<ValueT>* p_from, const Neuron<ValueT>* p_to)
	{
		//return std::make_pair(const_cast<Neuron<ValueT>*>(p_from), const_cast<Neuron<ValueT>*>(p_to));
		return std::make_pair(p_from, p_to);
	}


	private: FL_unique_ptr< Layer<ValueT> > p_inLayer_; /// Pointer to the input layer
    private: std::vector<Layer<ValueT>*> hiddenLayers_; ///< Collection of hidden layers
	private: FL_unique_ptr< Layer<ValueT> > p_outLayer_; /// Pointer to the output layer
    private: ConnectionMap conns_; ///< Connections between all neurons
}; // Network

}} // Namespace fl::ann

#endif // FL_ANN_NETWORKS_H
