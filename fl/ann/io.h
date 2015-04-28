/**
 * \file fl/ann/io.h
 *
 * \brief Input/output functions
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

#ifndef FL_ANN_IO_H
#define FL_ANN_IO_H


#include <algorithm>
#include <iomanip>
#include <iostream>
#include <my/commons.h>
#include <my/ann/layers.h>
#include <my/ann/networks.h>
#include <vector>


namespace fl { namespace ann {

template <typename CharT, typename CharTraitsT, typename ValueT>
std::basic_ostream<CharT,CharTraitsT>& operator<<(std::basic_ostream<CharT,CharTraitsT>& os, const Layer<ValueT>& layer)
{
	os << "Number of neurons: " << layer.numOfNeurons() << std::endl;
	os << "Has bias: " << std::boolalpha << layer.hasBias() << std::endl;

	return os;
}

template <typename CharT, typename CharTraitsT, typename ValueT>
std::basic_ostream<CharT,CharTraitsT>& operator<<(std::basic_ostream<CharT,CharTraitsT>& os, const Network<ValueT>& net)
{
	os	<< "Number of inputs: " << (net.getInputLayer() ? net.getInputLayer()->numOfNeurons() : 0) << std::endl
		<< "Number of outputs: " << (net.getOutputLayer() ? net.getOutputLayer()->numOfNeurons() : 0) << std::endl
		<< "Number of hidden layers: " << net.numOfHiddenLayers() << std::endl;
	for (std::size_t l = 0; l < net.numOfHiddenLayers(); ++l)
	{
		const std::size_t idx = l+1;
		Layer<ValueT>* p_layer = net.getLayer(idx);

		// check: null
		FL_DEBUG_ASSERT( p_layer );

		os << "Hidden layer #" << idx << ": " << *p_layer << std::endl;
	}   
	return os;
}

template <typename CharT, typename CharTraitsT, typename ValueT>
void DumpWeightsAndBiases(std::basic_ostream<CharT,CharTraitsT>& os, const Network<ValueT>& nnet)
{
	os << "[";
	for (std::size_t i = 1; i < nnet.numOfLayers(); ++i)
	{
		os << "(layer #" << i << ") [";
		Layer<ValueT>* p_layer = nnet.getLayer(i);
		for (typename Layer<ValueT>::ConstNeuronIterator neuronIt = p_layer->neuronBegin(),
																  neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;
			const std::vector<ValueT> weights = p_neuron->weights();
			std::copy(weights.begin(), weights.end(), std::ostream_iterator<double>(os, ", "));
			if (p_neuron->hasBias())
			{
				os << "{" << p_neuron->getBias() << "}, ";
			}
		}
		os << "], ";
	}
	os << "]";
}

template <typename CharT, typename CharTraitsT, typename ValueT>
void DumpNetInputs(std::basic_ostream<CharT,CharTraitsT>& os, const Network<ValueT>& nnet)
{
	os << "[";
	for (std::size_t i = 1; i < nnet.numOfLayers(); ++i)
	{
		os << "(layer #" << i << ") [";
		Layer<ValueT>* p_layer = nnet.getLayer(i);
		for (typename Layer<ValueT>::ConstNeuronIterator neuronIt = p_layer->neuronBegin(),
																  neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;
			os << p_neuron->getNetInput() << ", ";
		}
		os << "], ";
	}
	os << "]";
}


template <typename CharT, typename CharTraitsT, typename ValueT>
void DumpActivations(std::basic_ostream<CharT,CharTraitsT>& os, const Network<ValueT>& nnet)
{
	os << "[";
	for (std::size_t i = 1; i < nnet.numOfLayers(); ++i)
	{
		os << "(layer #" << i << ") [";
		Layer<ValueT>* p_layer = nnet.getLayer(i);
		for (typename Layer<ValueT>::ConstNeuronIterator neuronIt = p_layer->neuronBegin(),
																  neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			Neuron<ValueT>* p_neuron = *neuronIt;
			os << p_neuron->getOutput() << ", ";
		}
		os << "], ";
	}
	os << "]";
}

}} // Namespace fl::ann

#endif // FL_ANN_IO_H
