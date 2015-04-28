/**
 * \file test_ann.cpp
 *
 * \brief Test suite for artificial neural networks
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

#include <algorithm>
#include <iostream>
#include <fl/fuzzylite.h>
#include <fl/ann/activation_functions.h>
#include <fl/ann/error_functions.h>
#include <fl/ann/io.h>
#include <fl/ann/layers.h>
#include <fl/ann/net_input_functions.h>
#include <fl/ann/networks.h>
#include <fl/ann/neurons.h>
#include <fl/ann/training_algorithms.h>
#include <fl/ann/weight_randomizers.h>
#include <fl/dataset.h>
#ifdef FL_CPP11
# include <random>
#else // FL_CPP11
# include <boost/random.hpp>
#endif // FL_CPP11
#include <vector>


namespace detail {

template <typename ValueT>
fl::DataSet<ValueT> MakeDataSet()
{
	const std::size_t ni = 2;
	const std::size_t no = 1;
	const std::size_t nd = 4;
	double inputs[][ni] =  {{0, 0},
							{0, 1},
							{1, 0},
							{1, 1}};
	double outputs[][no] = {{0},
							{1},
							{1},
							{0}};

	fl::DataSet<ValueT> ds(ni,no);
	for (std::size_t i = 0; i < nd; ++i)
	{
		fl::DataSetEntry<ValueT> dse(inputs[i], inputs[i]+ni, outputs[i], outputs[i]+no);
		ds.add(dse);
	}

	return ds;
}

template <typename ValueT>
ValueT MaxDataSetNorm(const fl::DataSet<ValueT>& ds)
{
	ValueT maxNorm = 0;

	for (typename fl::DataSet<double>::ConstEntryIterator entryIt = ds.entryBegin(),
														  entryEndIt = ds.entryEnd();
		 entryIt != entryEndIt;
		 ++entryIt)
	{
		const fl::DataSetEntry<ValueT>& entry = *entryIt;

		ValueT norm = 0;
		for (typename fl::DataSetEntry<ValueT>::ConstInputIterator inputIt = entry.inputBegin(),
															  inputEndIt = entry.inputEnd();
			 inputIt != inputEndIt;
			 ++inputIt)
		{
			const ValueT x = *inputIt;
			norm += x*x;
		}
		norm = std::sqrt(norm);

		if (norm > maxNorm)
		{
			maxNorm = norm;
		}
	}

	return maxNorm;
}

template <typename CharT, typename CharTraitsT, typename ValueT>
void DumpWeightsAndBiases(std::basic_ostream<CharT,CharTraitsT>& os, const fl::ann::Network<ValueT>& nnet)
{
	os << "[";
	for (std::size_t i = 1; i < nnet.numOfLayers(); ++i)
	{
		os << "(layer #" << i << ") [";
		fl::ann::Layer<ValueT>* p_layer = nnet.getLayer(i);
		for (typename fl::ann::Layer<ValueT>::ConstNeuronIterator neuronIt = p_layer->neuronBegin(),
																  neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			fl::ann::Neuron<ValueT>* p_neuron = *neuronIt;
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
void DumpActivations(std::basic_ostream<CharT,CharTraitsT>& os, const fl::ann::Network<ValueT>& nnet)
{
	os << "[";
	for (std::size_t i = 1; i < nnet.numOfLayers(); ++i)
	{
		os << "(layer #" << i << ") [";
		fl::ann::Layer<ValueT>* p_layer = nnet.getLayer(i);
		for (typename fl::ann::Layer<ValueT>::ConstNeuronIterator neuronIt = p_layer->neuronBegin(),
																  neuronEndIt = p_layer->neuronEnd();
			 neuronIt != neuronEndIt;
			 ++neuronIt)
		{
			fl::ann::Neuron<ValueT>* p_neuron = *neuronIt;
			os << p_neuron->getOutput() << ", ";
		}
		os << "], ";
	}
	os << "]";
}

template <typename ValueT, typename IterT>
FL_unique_ptr< fl::ann::Network<ValueT> > MakeFeedforwardNetwork(std::size_t numInputs, IterT numHiddenFirst, IterT numHiddenLast, std::size_t numOutputs)
{
	FL_unique_ptr< fl::ann::Network<ValueT> > p_nnet(new fl::ann::Network<ValueT>());

	fl::ann::Layer<ValueT>* p_layer;
	fl::ann::Neuron<ValueT>* p_neuron;

	p_layer = new fl::ann::Layer<ValueT>(p_nnet.get());
	for (std::size_t i = 0; i < numInputs; ++i)
	{
		p_layer->addNeuron(new fl::ann::InputNeuron<ValueT>());
	}
	p_nnet->setInputLayer(p_layer);

	while (numHiddenFirst != numHiddenLast)
	{
		const std::size_t numHiddenUnits = *numHiddenFirst;

		fl::ann::Layer<ValueT>* p_prevLayer = p_layer;

		p_layer = new fl::ann::Layer<ValueT>(p_nnet.get());
		for (std::size_t i = 0; i < numHiddenUnits; ++i)
		{
			p_layer->addNeuron(
				new fl::ann::Neuron<ValueT>(
					new fl::ann::WeightedSumNetInputFunction<ValueT>(),
					new fl::ann::TanSigmoidActivationFunction<ValueT>()
					//new fl::ann::LogSigmoidActivationFunction<ValueT>()
				)
			);
		}
		p_layer->setHasBias(true);
		p_nnet->addHiddenLayer(p_layer);

		p_nnet->connect(p_prevLayer, p_layer, 0);

		++numHiddenFirst;
	}

	fl::ann::Layer<ValueT>* p_prevLayer = p_layer;
	p_layer = new fl::ann::Layer<ValueT>(p_nnet.get());
	for (std::size_t i = 0; i < numOutputs; ++i)
	{
		p_layer->addNeuron(
			new fl::ann::Neuron<ValueT>(
				new fl::ann::WeightedSumNetInputFunction<ValueT>(),
				new fl::ann::TanSigmoidActivationFunction<ValueT>()
			)
		);
	}
	p_layer->setHasBias(true);
	p_nnet->setOutputLayer(p_layer);
	p_nnet->connect(p_prevLayer, p_layer, 0);

	return p_nnet;
}

template <typename ValueT>
FL_unique_ptr< fl::ann::Network<ValueT> > MakeFeedforwardNetwork(std::size_t numInputs, std::size_t numHiddenLayers, std::size_t numNeuronsPerHiddenLayer, std::size_t numOutputs)
{
	std::vector<std::size_t> numHiddens(numHiddenLayers, numNeuronsPerHiddenLayer);

	return MakeFeedforwardNetwork<ValueT>(numInputs, numHiddens.begin(), numHiddens.end(), numOutputs);
}

} // Namespace detail


int main()
{
    const std::size_t numInputs = 2;
    const std::size_t numOutputs = 1;
    //const std::size_t numLayers = 2;
    const std::size_t numLayersHidden = 1;
    const std::size_t numNeuronsHidden = 3;
    const double maxError = 0.01f;
    const std::size_t maxEpochs = 5000;

	FL_unique_ptr< fl::ann::Network<double> > p_nnet = detail::MakeFeedforwardNetwork<double>(numInputs, numLayersHidden, numNeuronsHidden, numOutputs);

	std::cout << "Neural Network" << std::endl
			  << "--------------" << std::endl
			  << (*p_nnet) << std::endl
			  << "--------------" << std::endl;

#ifdef FL_CPP11
	std::default_random_engine rng;
#else // FL_CPP11
	boost::random::mt19937 rng;
#endif // FL_CPP11

	fl::DataSet<double> ds = detail::MakeDataSet<double>();

	//fl::ann::ConstWeightRandomizer<double> nwrand(0.05);
	//fl::ann::RangeWeightRandomizer<double> nwrand(-0.05, 0.05);
	fl::ann::RangeWeightRandomizer<double> nwrand(-1, 1);
	//fl::ann::NguyenWidrowWeightRandomizer<double> nwrand;
	//fl::ann::GaussianWeightRandomizer<double> nwrand(detail::MaxDataSetNorm(ds));
	nwrand.randomize(*p_nnet, rng);
	std::cout << "Weights & Biases randomization: ";
	detail::DumpWeightsAndBiases(std::cout, *p_nnet);
	std::cout << std::endl;

	fl::ann::GradientDescentBackpropagationAlgorithm<double> trainer;
	trainer.setLearningRate(0.01);
	trainer.setMomentum(0);
	trainer.setNetwork(p_nnet.get());
	//trainer.setErrorFunction(new fl::ann::SumSquaredErrorFunction<double>(0.5));
	trainer.setErrorFunction(new fl::ann::MeanSquaredErrorFunction<double>());
	double trainErr = 0;
	std::size_t epoch = 0;
	do
	{
		++epoch;

		std::cout << "EPOCH #" << epoch << std::endl;

		trainErr = trainer.trainSingleEpoch(ds);

		std::cout << "  -> Weights & Biases: ";
		detail::DumpWeightsAndBiases(std::cout, *p_nnet);
		std::cout << std::endl;
		std::cout << "  -> Activations: ";
		detail::DumpActivations(std::cout, *p_nnet);
		std::cout << std::endl;
		std::cout << "  -> Error: " << trainErr << std::endl;
	}
	while (trainErr > maxError && epoch < maxEpochs);

	std::cout << "Trained Network: " << std::endl;
	std::cout << "  Weights: ";
	detail::DumpWeightsAndBiases(std::cout, *p_nnet);
	std::cout << std::endl;

	std::cout << "Testing network..." << std::endl;
	for (typename fl::DataSet<double>::ConstEntryIterator entryIt = ds.entryBegin(),
														  entryEndIt = ds.entryEnd();
		 entryIt != entryEndIt;
		 ++entryIt)
	{
		const std::vector<double> inputs(entryIt->inputBegin(), entryIt->inputEnd());
		const std::vector<double> targetOuts(entryIt->outputBegin(), entryIt->outputEnd());

		const std::vector<double> actualOuts = p_nnet->process(inputs.begin(), inputs.end());

		std::cout << "Input: (" << inputs[0] << "," << inputs[1] << ") -> " << actualOuts[0] << ", should be " << targetOuts[0] << ", difference " << std::abs(actualOuts[0]-targetOuts[0]) << std::endl;
	}

	std::vector<double> input(numInputs);
	//input[0] = -1;
	//input[1] = +1;
	input[0] = 1;
	input[1] = 0;

	std::vector<double> out = p_nnet->process(input.begin(), input.end());
	for (std::size_t i = 0; i < numOutputs; ++i)
	{
		std::cout << "output[" << i << "] => " << out[i] << std::endl;
	}
}
