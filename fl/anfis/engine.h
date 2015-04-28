/**
 * \file fl/anfis/engine.h
 *
 * \brief The ANFIS engine class
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

#ifndef FL_ANFIS_ENGINE_H
#define FL_ANFIS_ENGINE_H


#include <fl/anfis/nodes.h>
#include <fl/Engine.h>
#include <fl/fuzzylite.h>
#include <map>
#include <string>
#include <vector>


namespace fl { namespace anfis {

class FL_API Engine: public fl::Engine
{
private:
	typedef fl::Engine BaseType;

public:
	enum LayerCategory
	{
		InputLayer,
		FuzzificationLayer,
		InputHedgeLayer,
		AntecedentLayer,
		ConsequentLayer,
		AccumulationLayer,
		OutputLayer
	};


	explicit Engine(const std::string& name = "");

	template <typename InputIterT,
			  typename OutputIterT,
			  typename RuleBlockIterT>
	Engine(InputIterT inputFirst, InputIterT inputLast,
		   OutputIterT outputFirst, OutputIterT outputLast,
		   RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast,
		   const std::string& name = "");

	explicit Engine(const fl::Engine& other);

	Engine(const Engine& other);

	FL_DEFAULT_MOVE(Engine)

	virtual ~Engine();

	Engine& operator=(const Engine& rhs);

	Engine* clone() const;

	template <typename IterT>
	void setInputVariables(IterT first, IterT last);

	template <typename IterT>
	void setOutputVariables(IterT first, IterT last);

	template <typename IterT>
	void setRuleBlocks(IterT first, IterT last);

	bool isReady(std::string* status = fl::null) const;

	void process();

	void restart();

	void build();

	template <typename InputIterT,
			  typename OutputIterT,
			  typename RuleBlockIterT>
	void build(InputIterT inputFirst, InputIterT inputLast,
			   OutputIterT outputFirst, OutputIterT outputLast,
			   RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast);

	void clear();

	void setIsLearning(bool value);

	bool isLearning() const;

	void setHasBias(bool value);

	bool hasBias() const;

	template <typename IterT>
	void setBias(IterT first, IterT last);

	void setBias(const std::vector<fl::scalar>& value);

	std::vector<fl::scalar> getBias() const;

	std::vector<Node*> inputConnections(const Node* p_node) const;

	std::vector<Node*> outputConnections(const Node* p_node) const;

	template <typename IterT>
	void setInputValues(IterT first, IterT last);

	std::vector<fl::scalar> getInputValues() const;

	std::vector<InputNode*> getInputLayer() const;

	std::vector<FuzzificationNode*> getFuzzificationLayer() const;

	std::vector<InputHedgeNode*> getInputHedgeLayer() const;

	std::vector<AntecedentNode*> getAntecedentLayer() const;

	std::vector<ConsequentNode*> getConsequentLayer() const;

	std::vector<AccumulationNode*> getAccumulationLayer() const;

	std::vector<OutputNode*> getOutputLayer() const;

	std::vector<Node*> getLayer(LayerCategory layer) const;

	std::vector<fl::scalar> eval();

	std::vector<fl::scalar> evalTo(LayerCategory layer);

	std::vector<fl::scalar> evalFrom(LayerCategory layer);

	template <typename IterT>
	std::vector<fl::scalar> eval(IterT first, IterT last);

	template <typename IterT>
	std::vector<fl::scalar> evalTo(IterT first, IterT last, LayerCategory layer);

	LayerCategory getNextLayerCategory(LayerCategory cat) const;

	LayerCategory getPreviousLayerCategory(LayerCategory cat) const;

protected:
	void updateAnfisReferences();

private:
	void check();

	void clearAnfis();

	void connect(Node* p_from, Node* p_to);

	std::vector<fl::scalar> evalLayer(LayerCategory layer);

	template <typename IterT>
	std::vector<fl::scalar> evalLayer(IterT first, IterT last);

	std::vector<fl::scalar> evalInputLayer();

	std::vector<fl::scalar> evalFuzzificationLayer();

	std::vector<fl::scalar> evalInputHedgeLayer();

	std::vector<fl::scalar> evalAntecedentLayer();

	std::vector<fl::scalar> evalConsequentLayer();

	std::vector<fl::scalar> evalAccumulationLayer();

	std::vector<fl::scalar> evalOutputLayer();

private:
	std::vector<InputNode*> inputNodes_; ///< Nodes in the input layer
	std::vector<FuzzificationNode*> fuzzificationNodes_; ///< Nodes in the fuzzification layer for linguistic terms
	std::vector<InputHedgeNode*> inputHedgeNodes_; ///< Additional nodes in the fuzzification layer for linguistic hedges
	std::vector<AntecedentNode*> antecedentNodes_; ///< Nodes in the antecedent layer
	std::vector<ConsequentNode*> consequentNodes_; ///< Nodes in the consequent layer
	std::vector<AccumulationNode*> accumulationNodes_; ///< Nodes in the summation layer
	std::vector<OutputNode*> outputNodes_; ///< Nodes in the inference layer
	std::map< const Node*, std::vector<Node*> > inConns_; ///< Input connection to a given node
	std::map< const Node*, std::vector<Node*> > outConns_; ///< Output connection from a given node
	bool hasBias_; ///< If \c true, the bias vector is used in place of the output values in case of zero firing strength
	bool isLearning_; ///< \c true if the ANFIS is in the learning modality
}; // Engine

}} // Namespace fl::anfis


#include "fl/anfis/engine.tpp"


#endif // FL_ANFIS_ENGINE_H
