#ifndef FL_ANFIS_MODEL_H
#define FL_ANFIS_MODEL_H

#include <boost/noncopyable.hpp>
#include <fl/commons.h>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/Headers.h>


namespace fl { namespace anfis {

class Model;


////////////////////////////////////////////////////////////////////////////////
// Nodes
////////////////////////////////////////////////////////////////////////////////


class Node: boost::noncopyable
{
public:
	Node(Model* p_model);

	virtual ~Node();

	void setModel(Model* p_model);

	Model* getModel() const;

	std::vector<Node*> inputConnections() const;

	std::vector<Node*> outputConnections() const;

	std::vector<fl::scalar> inputs() const;

	fl::scalar eval();

	fl::scalar getValue() const;

protected:
	void setValue(fl::scalar v);

private:
	virtual fl::scalar doEval() = 0;


private:
	Model* p_model_;
	fl::scalar val_;
}; // Node

class InputNode: public Node
{
public:
	InputNode(fl::InputVariable* p_var, Model* p_model);

	fl::InputVariable* getInputVariable() const;

private:
	fl::scalar doEval();


private:
	fl::InputVariable* p_var_;
}; // InputNode

class TermNode: public Node
{
public:
	TermNode(fl::Term* p_term, Model* p_model);

	fl::Term* getTerm() const;

private:
	fl::scalar doEval();


private:
	fl::Term* p_term_;
}; // TermNode

class HedgeNode: public Node
{
public:
	HedgeNode(fl::Hedge* p_hedge, Model* p_model);

	fl::Hedge* getHedge() const;

private:
	fl::scalar doEval();


private:
	fl::Hedge* p_hedge_;
}; // HedgeNode

class RuleFiringStrengthNode: public Node
{
public:
	RuleFiringStrengthNode(fl::Norm* p_norm, Model* p_model);

	fl::Norm* getNorm() const;

private:
	fl::scalar doEval();

/*
	std::vector<fl::scalar> doEvalDerivativeWrtInput(ForwardIterator inputFirst, ForwardIterator inputLast,
															  ForwardIterator weightFirst, ForwardIterator weightLast) const;
*/


private:
	fl::Norm* p_norm_;
}; // AntecedentNode


class RuleImplicationNode: public Node
{
public:
	RuleImplicationNode(fl::Term* p_term, fl::TNorm* p_tnorm, Model* p_model);

	fl::Term* getTerm() const;

	fl::TNorm* getTNorm() const;

private:
	fl::scalar doEval();


private:
	fl::Term* p_term_;
	fl::TNorm* p_tnorm_;
};

class SumNode: public Node
{
public:
	explicit SumNode(Model* p_model);

private:
	fl::scalar doEval();
}; // SumNode

class NormalizationNode: public Node
{
public:
	explicit NormalizationNode(Model* p_model);

private:
	fl::scalar doEval();
}; // NormalizationNode

class Model
{
public:
	Model();

	template <typename InputIterT,
			  typename OutputIterT,
			  typename RuleBlockIterT>
	Model(InputIterT inputFirst, InputIterT inputLast,
		  OutputIterT outputFirst, OutputIterT outputLast,
		  RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast)
	: inputs_(inputFirst, inputLast),
	  outputs_(outputFirst, outputLast),
	  ruleBlocks_(ruleBlockFirst, ruleBlockLast)
//	  order_(0)
	{
		this->build();
	}

	~Model();

	void build();

	template <typename InputIterT,
			  typename OutputIterT,
			  typename RuleBlockIterT>
	void build(InputIterT inputFirst, InputIterT inputLast,
			   OutputIterT outputFirst, OutputIterT outputLast,
			   RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast)
	{
		this->clear();

		inputs_.assign(inputFirst, inputLast);
		outputs_.assign(outputFirst, outputLast);
		ruleBlocks_.assign(ruleBlockFirst, ruleBlockLast);

		this->build();
	}

	std::vector<Node*> inputConnections(const Node* p_node) const;

	std::vector<Node*> outputConnections(const Node* p_node) const;

	template <typename IterT>
	void setInput(IterT first, IterT last)
	{
		std::vector<InputNode*>::iterator nodeIt = inputNodes_.begin();
		std::vector<InputNode*>::iterator nodeEndIt = inputNodes_.end();

		while (first != last && nodeIt != nodeEndIt)
		{
			(*nodeIt)->getInputVariable()->setInputValue(*first);

			++first;
			++nodeIt;
		}

		if (first != last || nodeIt != nodeEndIt)
		{
			FL_THROW2(std::invalid_argument, "Wrong number of inputs");
		}
	}

	std::vector<fl::scalar> eval();

	template <typename IterT>
	std::vector<fl::scalar> eval(IterT first, IterT last)
	{
		this->setInput(first, last);
		return this->eval();
	}

	void clear();

private:
	void check();

	void connect(Node* p_from, Node* p_to);

	template <typename IterT>
	std::vector<fl::scalar> evalLayer(IterT first, IterT last)
	{
		std::vector<fl::scalar> res;
		while (first != last)
		{
			Node* p_node = *first;

			FL_DEBUG_ASSERT( p_node );

			res.push_back(p_node->eval());

			++first;
		}
		return res;
	}

	std::vector<fl::scalar> evalInputLayer();

	std::vector<fl::scalar> evalFuzzificationLayer();

	std::vector<fl::scalar> evalHedgeLayer();

	std::vector<fl::scalar> evalAntecedentLayer();

	std::vector<fl::scalar> evalConsequentLayer();

	std::vector<fl::scalar> evalAccumulationLayer();

	std::vector<fl::scalar> evalNormalizationLayer();

private:
	std::vector<fl::InputVariable*> inputs_; ///< Collection of (pointer to) input variables
	std::vector<fl::OutputVariable*> outputs_; ///< Collection of (pointer to) output variables
	std::vector<fl::RuleBlock*> ruleBlocks_; ///< Collection of (pointer to) rule blocks
	//std::size_t order_;
	std::vector<InputNode*> inputNodes_; ///< Nodes in the input layer
	std::vector<TermNode*> inputTermNodes_; ///< Nodes in the fuzzification layer for linguistic terms
	std::vector<HedgeNode*> inputHedgeNodes_; ///< Additional nodes in the fuzzification layer for linguistic hedges
	std::vector<RuleFiringStrengthNode*> antecedentNodes_; ///< Nodes in the antecedent layer
	std::vector<RuleImplicationNode*> consequentNodes_; ///< Nodes in the consequent layer
	std::vector<SumNode*> sumNodes_; ///< Nodes in the summation layer
	std::vector<NormalizationNode*> inferenceNodes_; ///< Nodes in the inference layer
	std::map< const Node*, std::vector<Node*> > inConns_; ///< Input connection to a given node
	std::map< const Node*, std::vector<Node*> > outConns_; ///< Output connection from a given node
}; // Model


}} // Namespace fl::anfis

#endif // FL_ANFIS_MODEL_H
