/**
 * \file fl/ann/neurons.h
 *
 * \brief Neurons for artificial neural networks
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

#ifndef FL_ANN_NEURONS_H
#define FL_ANN_NEURONS_H


#include <fl/fuzzylite.h>
#include <limits>
#include <my/ann/activation_functions.h>
#include <my/ann/net_input_functions.h>
#include <my/commons.h>
#include <vector>


namespace fl { namespace ann {

template <typename T> class Connection;
template <typename T> class Layer;

/**
 * General neuron model according to McCulloch-Pitts artificial neuron model.
 *
 * The artificial neuron is the basic processing unit of a neural network.
 * The artificial neuron receives one or more inputs and one or more weights (one for each input), and produces an output.
 * The inputs and weights are combined together by means of the <em>input function</em> (or <em>propagation function</em>) and the result is passed through a non-linear function known as an <em>activation function</em> (or <em>transfer function</em>).
 * The input function usually is the sum of the product of each input with the corresponding weigth, but other input functions are possible.
 * The activation function usually has a sigmoid shape, but it may also take the form of other non-linear function (e.g., piecewise linear function or step function, just to name a few).
 * Also, it is often monotonically increasing, continuous, differentiable and bounded.
 */

template <typename ValueT>
class Neuron
{
//	private: typedef std::vector<Connection<ValueT>*> ConnectionContainer;
	public: typedef ValueT ValueType;


	/**
	 * Creates an artificial neuron with the weighted-sum function as input function 
	 * and the step function as activation function.
	 * This is the original McCulloch-Pitts neuron model.
	 */
	public: explicit Neuron(Layer<ValueT>* p_layer = fl::null)
	: p_inpFunc_(new WeightedSumNetInputFunction<ValueT>()),
	  p_actFunc_(new StepActivationFunction<ValueT>()),
	  p_layer_(p_layer),
	  netIn_(std::numeric_limits<ValueT>::quiet_NaN()),
	  out_(std::numeric_limits<ValueT>::quiet_NaN()),
//	  err_(std::numeric_limits<ValueT>::quiet_NaN()),
	  hasBias_(false),
	  biasWeight_(0)
	{
//FL_DEBUG_TRACE("In Neuron's constructor (" << this << ")");//XXX
	}

	/// Creates an artificial neuron with the given input and activation functions
	public: Neuron(NetInputFunction<ValueT>* p_inpFunc, ActivationFunction<ValueT>* p_actFunc, Layer<ValueT>* p_layer = fl::null)
	: p_inpFunc_(p_inpFunc),
	  p_actFunc_(p_actFunc),
	  p_layer_(p_layer),
	  netIn_(std::numeric_limits<ValueT>::quiet_NaN()),
	  out_(std::numeric_limits<ValueT>::quiet_NaN()),
//	  err_(std::numeric_limits<ValueT>::quiet_NaN()),
	  hasBias_(false),
	  biasWeight_(0)
	{
//FL_DEBUG_TRACE("In Neuron's constructor (" << this << ")");//XXX
		if (!p_inpFunc_.get())
		{
			FL_THROW2(std::invalid_argument, "Input function cannot be null");
		}
		if (!p_actFunc_.get())
		{
			FL_THROW2(std::invalid_argument, "Activation function cannot be null");
		}
	}

	public: virtual ~Neuron()
	{
		this->clear();
	}

	public: void clear()
	{
		this->reset();

		this->clearInputConnections();

		this->clearOutputConnections();

		p_inpFunc_.reset();

		p_actFunc_.reset();

		p_layer_ = fl::null;
	}

	public: void reset()
	{
//FL_DEBUG_TRACE("In Neuron's reset (" << this << ")");//XXX
		netIn_ = out_
//			   = err_
			   = std::numeric_limits<ValueT>::quiet_NaN();

		hasBias_ = false;
		biasWeight_ = 0;

		// Reset output connections
		std::vector<Connection<ValueT>*> outConns = this->outputConnections();
		for (typename std::vector<Connection<ValueT>*>::iterator connIt = outConns.begin(),
																 connEndIt = outConns.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			Connection<ValueT>* p_conn = *connIt;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			p_conn->reset();
		}
	}

	public: void setNetInputFunction(NetInputFunction<ValueT>* p_func)
	{
		if (!p_func)
		{
			FL_THROW2(std::invalid_argument, "Input function cannot be null");
		}

		p_inpFunc_.reset(p_func);
	}

	public: NetInputFunction<ValueT>* getNetInputFunction() const
	{
		return p_inpFunc_.get();
	}

	public: void setActivationFunction(ActivationFunction<ValueT>* p_func)
	{
		if (!p_func)
		{
			FL_THROW2(std::invalid_argument, "Activation function cannot be null");
		}

		p_actFunc_.reset(p_func);
	}

	public: ActivationFunction<ValueT>* getActivationFunction() const
	{
		return p_actFunc_.get();
	}

	public: void setLayer(Layer<ValueT>* p_layer)
	{
		p_layer_ = p_layer;
	}

	public: Layer<ValueT>* getLayer() const
	{
		return p_layer_;
	}

	public: void addInputConnection(Neuron<ValueT>* p_fromNeuron, ValueT weight)
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		p_layer_->getNetwork()->connect(p_fromNeuron, this, weight);
	}

	public: void eraseInputConnection(Neuron<ValueT>* p_fromNeuron)
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		p_layer_->getNetwork()->disconnect(p_fromNeuron, this);
	}

	public: std::vector<Connection<ValueT>*> inputConnections() const
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		return p_layer_->getNetwork()->inputConnections(this);
	}

	public: void clearInputConnections()
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		p_layer_->getNetwork()->disconnectTo(this);
	}

	public: void addOutputConnection(Neuron<ValueT>* p_toNeuron, ValueT weight)
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		p_layer_->getNetwork()->connect(this, p_toNeuron, weight);
	}

	public: void eraseOutputConnection(Neuron<ValueT>* p_toNeuron)
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		p_layer_->getNetwork()->disconnect(this, p_toNeuron);
	}

	public: std::vector<Connection<ValueT>*> outputConnections() const
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		return p_layer_->getNetwork()->outputConnections(this);
	}

	public: void clearOutputConnections()
	{
		FL_DEBUG_ASSERT( p_layer_ );
		FL_DEBUG_ASSERT( p_layer_->getNetwork() );

		p_layer_->getNetwork()->disconnectFrom(this);
	}

	/// Sets the net input for this neuron
	public: void setNetInput(ValueT v)
	{
		netIn_ = v;
	}

	/// Gets the net input of this neuron
	public: ValueT getNetInput() const
	{
		return netIn_;
	}

	/// Returns the input values to this neuron
	public: std::vector<ValueT> inputs() const
	{
		std::vector<ValueT> ins;

		const std::vector<Connection<ValueT>*> inConns = this->inputConnections();
		for (typename std::vector<Connection<ValueT>*>::const_iterator connIt = inConns.begin(),
																	   connEndIt = inConns.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			const Connection<ValueT>* p_conn = *connIt;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			ins.push_back(p_conn->getFromNeuron()->getOutput());
		}

		return ins;
	}

	/// Sets the weight of this neuron's input connections to the values specified by the iterator range [\a first, \a last).
	public: template <typename IterT>
			void weights(IterT weightFirst, IterT weightLast)
	{
		const std::vector<Connection<ValueT>*> inConns = this->inputConnections();
		for (typename std::vector<Connection<ValueT>*>::iterator connIt = inConns.begin(),
																 connEndIt = inConns.end();
			 connIt != connEndIt && weightFirst != weightLast;
			 ++connIt)
		{
			Connection<ValueT>* p_conn = *connIt;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			p_conn->setWeight(*weightFirst);

			++weightFirst;
		}

	}

	/// Returns the weight values to this neuron
	public: std::vector<ValueT> weights() const
	{
		std::vector<ValueT> ws;

		const std::vector<Connection<ValueT>*> inConns = this->inputConnections();
		for (typename std::vector<Connection<ValueT>*>::const_iterator connIt = inConns.begin(),
																	   connEndIt = inConns.end();
			 connIt != connEndIt;
			 ++connIt)
		{
			const Connection<ValueT>* p_conn = *connIt;

			// check: null
			FL_DEBUG_ASSERT( p_conn );

			ws.push_back(p_conn->getWeight());
		}

		return ws;
	}

	/// Calculates neuron's output
	public: virtual ValueT process()
	{
		// check: null
		FL_DEBUG_ASSERT( p_inpFunc_.get() );
		FL_DEBUG_ASSERT( p_actFunc_.get() );

		std::vector<ValueT> ins = this->inputs();
		std::vector<ValueT> ws = this->weights();
		if (hasBias_)
		{
			ins.push_back(1);
			ws.push_back(biasWeight_);
		}
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "(neuron: " << this << ") Inputs: [";
//std::copy(ins.begin(), ins.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]," << std::endl;
//std::cerr << "(neuron: " << this << ") Weights: [";
//std::copy(ws.begin(), ws.end(), std::ostream_iterator<ValueT>(std::cerr, ", "));
//std::cerr << "]" << std::endl;
//#endif // FL_DEBUG
////[/XXX]

		netIn_ = p_inpFunc_->eval(ins.begin(), ins.end(), ws.begin(), ws.end());
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "(neuron: " << this << ") net input: " << netIn_ << std::endl;
//#endif // FL_DEBUG
////[/XXX]

		out_ = p_actFunc_->eval(netIn_);
////[XXX]
//#ifdef FL_DEBUG
//std::cerr << "(neuron: " << this << ") output: " << out_ << std::endl;
//#endif // FL_DEBUG
////[/XXX]

		return out_;
	}

	public: ValueT getOutput() const
	{
		return out_;
	}

	protected: void setOutput(ValueT v)
	{
		out_ = v;
	}

//	public: void setError(ValueT v)
//	{
//		err_ = v;
//	}

//	public: ValueT getError() const
//	{
//		return err_;
//	}

	public: void setHasBias(bool v)
	{
		hasBias_ = v;
	}

	public: bool hasBias() const
	{
		return hasBias_;
	}

	public: void setBias(ValueT wb)
	{
		biasWeight_ = wb;
	}

	public: ValueT getBias() const
	{
		return biasWeight_;
	}


	private: FL_unique_ptr< NetInputFunction<ValueT> > p_inpFunc_; ///< The input function
	private: FL_unique_ptr< ActivationFunction<ValueT> > p_actFunc_; ///< The activation function
	private: Layer<ValueT>* p_layer_; ///< The layer containing this neuron
	private: ValueT netIn_; ///< The input for this neuron received from the net input function
	private: ValueT out_; ///< The output of this neuron
//	private: ValueT err_; //< The error term
	private: bool hasBias_; ///< Tells if this neuron has a bias connection
	private: ValueT biasWeight_; ///< The weight associated to the bias connection
}; // Neuron


///**
// * A bias neuron is a neuron whose output value is always 1
// *
// * \author Marco Guazzone (marco.guazzone@gmail.com)
// */
//template <typename ValueT>
//class BiasNeuron: public Neuron<ValueT>
//{
//	private: typedef Neuron<ValueT> BaseType;
//
//	private: static const std::vector<Connection<ValueT>*> emptyConnections;
//
//
//	public: BiasNeuron(Layer<ValueT>* p_layer = fl::null)
//	: BaseType(new ConstantNetInputFunction<ValueT>(1), new LinearActivationFunction<ValueT>(1), p_layer) 
//	{
//FL_DEBUG_TRACE("In BiasNeuron's constructor (" << this << ")");//XXX
//		this->setNetInput(1);
//		this->process();
//	}
//
//	public: ValueT process()
//	{
//		const ValueT out = this->getActivationFunction()->eval(this->getNetInput());
//		this->setOutput(out);
//
//		return out;
//	}
//
//	public: bool isDummy() const
//	{
//		return true;
//	}
//}; // BiasNeuron


/**
 * A neuron that can be used in the input layer of a neural network
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class InputNeuron: public Neuron<ValueT>
{
	private: typedef Neuron<ValueT> BaseType;

	//private: static const std::vector<Connection<ValueT>*> emptyConnections;


	public: explicit InputNeuron(Layer<ValueT>* p_layer = fl::null)
	: BaseType(new WeightedSumNetInputFunction<ValueT>(), new PureLinearActivationFunction<ValueT>(), p_layer) 
	{
//FL_DEBUG_TRACE("In InputNeuron's constructor (" << this << ")");//XXX
	}

	public: void setInput(ValueT v)
	{
		this->setNetInput(v);
		this->process();
	}

	public: ValueT getInput() const
	{
		return this->getNetInput();
	}

	public: ValueT process()
	{
		const ValueT out = this->getActivationFunction()->eval(this->getInput());

		this->setOutput(out);

		return out;
	}
}; // InputNeuron

}} // Namespace fl::ann

#endif // FL_ANN_NEURONS_H
