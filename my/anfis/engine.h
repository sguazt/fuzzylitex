#ifndef FL_ANFIS_ENGINE_H
#define FL_ANFIS_ENGINE_H

#include <boost/noncopyable.hpp>
#include <cstddef>
#include <fl/commons.h>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/Headers.h>
#include <string>


namespace fl { namespace anfis {

class Engine;


////////////////////////////////////////////////////////////////////////////////
// Nodes
////////////////////////////////////////////////////////////////////////////////


class Node: boost::noncopyable
{
public:
	Node(Engine* p_engine);

	virtual ~Node();

	void setEngine(Engine* p_engine);

	Engine* getEngine() const;

	std::vector<Node*> inputConnections() const;

	std::vector<Node*> outputConnections() const;

	std::vector<fl::scalar> inputs() const;

	fl::scalar eval();

	std::vector<fl::scalar> evalDerivativeWrtInputs();

	fl::scalar getValue() const;

//protected:
	void setValue(fl::scalar v);

private:
	virtual fl::scalar doEval() = 0;

	virtual std::vector<fl::scalar> doEvalDerivativeWrtInputs() = 0;


private:
	Engine* p_engine_;
	fl::scalar val_;
}; // Node

class InputNode: public Node
{
public:
	InputNode(fl::InputVariable* p_var, Engine* p_engine);

	fl::InputVariable* getInputVariable() const;

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();


private:
	fl::InputVariable* p_var_;
}; // InputNode

class FuzzificationNode: public Node
{
public:
	FuzzificationNode(fl::Term* p_term, Engine* p_engine);

	fl::Term* getTerm() const;

	std::vector<fl::scalar> evalDerivativeWrtParams();

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();


private:
	fl::Term* p_term_;
}; // FuzzificationNode

class InputHedgeNode: public Node
{
public:
	InputHedgeNode(fl::Hedge* p_hedge, Engine* p_engine);

	~InputHedgeNode();

	fl::Hedge* getHedge() const;

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();


private:
	fl::Hedge* p_hedge_;
}; // InputHedgeNode

class AntecedentNode: public Node
{
public:
	AntecedentNode(fl::Norm* p_norm, Engine* p_engine);

	fl::Norm* getNorm() const;

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();


private:
	fl::Norm* p_norm_;
}; // AntecedentNode


class ConsequentNode: public Node
{
public:
	ConsequentNode(fl::Term* p_term, fl::TNorm* p_tnorm, Engine* p_engine);

	fl::Term* getTerm() const;

	fl::TNorm* getTNorm() const;

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();


private:
	fl::Term* p_term_;
	fl::TNorm* p_tnorm_;
};

class AccumulationNode: public Node
{
public:
	explicit AccumulationNode(Engine* p_engine);

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();
}; // AccumulationNode

class OutputNode: public Node
{
public:
	explicit OutputNode(Engine* p_engine);

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();
}; // OutputNode


class Engine: public fl::Engine
{
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
		   const std::string& name = "")
	: inputs_(inputFirst, inputLast),
	  outputs_(outputFirst, outputLast),
	  ruleBlocks_(ruleBlockFirst, ruleBlockLast),
	  name_(name)
//	  order_(0)
	{
		this->build();
	}

	Engine(const Engine& other);

	FL_DEFAULT_MOVE(Engine)

	virtual ~Engine();

	Engine& operator=(const Engine& rhs);

	Engine* clone() const;

	std::string toString() const;

	void configure(fl::TNorm* conjunction, fl::SNorm* disjunction,
				   fl::TNorm* activation, fl::SNorm* accumulation,
				   fl::Defuzzifier* defuzzifier);

	void configure(const std::string& conjunctionT,
				   const std::string& disjunctionS,
				   const std::string& activationT,
				   const std::string& accumulationS,
				   const std::string& defuzzifier,
				   int resolution = fl::IntegralDefuzzifier::defaultResolution());

	void setName(const std::string& name);

	std::string getName() const;

	void addInputVariable(fl::InputVariable* p_var);

	//fl::InputVariable* setInputVariable(fl::InputVariable* p_var, std::size_t idx);
	fl::InputVariable* setInputVariable(fl::InputVariable* p_var, int idx);

	//void insertInputVariable(fl::InputVariable* p_var, std::size_t idx);
	void insertInputVariable(fl::InputVariable* p_var, int idx);

	//fl::InputVariable* getInputVariable(std::size_t idx) const;
	fl::InputVariable* getInputVariable(int idx) const;

	fl::InputVariable* getInputVariable(const std::string& name) const;

	//fl::InputVariable* removeInputVariable(std::size_t idx);
	fl::InputVariable* removeInputVariable(int idx);

	fl::InputVariable* removeInputVariable(const std::string& name);

	bool hasInputVariable(const std::string& name) const;

	void setInputVariables(const std::vector<fl::InputVariable*>& vars);

	template <typename IterT>
	void setInputVariables(IterT first, IterT last)
	{
		inputs_.clear();
		inputs_.assign(first, last);
	}

	std::vector<fl::InputVariable*> const& inputVariables() const;

	//std::size_t numberOfInputVariables() const;
	int numberOfInputVariables() const;

	void addOutputVariable(fl::OutputVariable* p_var);

	//fl::OutputVariable* setOutputVariable(fl::OutputVariable* p_var, std::size_t idx);
	fl::OutputVariable* setOutputVariable(fl::OutputVariable* p_var, int idx);

	//void insertOutputVariable(fl::OutputVariable* p_var, std::size_t idx);
	void insertOutputVariable(fl::OutputVariable* p_var, int idx);

	//fl::OutputVariable* getOutputVariable(std::size_t idx) const;
	fl::OutputVariable* getOutputVariable(int idx) const;

	fl::OutputVariable* getOutputVariable(const std::string& name) const;

	//fl::OutputVariable* removeOutputVariable(std::size_t idx);
	fl::OutputVariable* removeOutputVariable(int idx);

	fl::OutputVariable* removeOutputVariable(const std::string& name);

	bool hasOutputVariable(const std::string& name) const;

	void setOutputVariables(const std::vector<fl::OutputVariable*>& vars);

	template <typename IterT>
	void setOutputVariables(IterT first, IterT last)
	{
		outputs_.clear();
		outputs_.assign(first, last);
	}

	std::vector<fl::OutputVariable*> const& outputVariables() const;

	//std::size_t numberOfOutputVariables() const;
	int numberOfOutputVariables() const;

	std::vector<fl::Variable*> variables() const;

	void addRuleBlock(fl::RuleBlock* p_block);

	//fl::RuleBlock* setRuleBlock(fl::RuleBlock* p_block, std::size_t idx);
	fl::RuleBlock* setRuleBlock(fl::RuleBlock* p_block, int idx);

	//void insertRuleBlock(fl::RuleBlock* p_block, std::size_t idx);
	void insertRuleBlock(fl::RuleBlock* p_block, int idx);

	//fl::RuleBlock* getRuleBlock(std::size_t idx) const;
	fl::RuleBlock* getRuleBlock(int idx) const;

	fl::RuleBlock* getRuleBlock(const std::string& name) const;

	bool hasRuleBlock(const std::string& name) const;

	//fl::RuleBlock* removeRuleBlock(std::size_t idx);
	fl::RuleBlock* removeRuleBlock(int idx);

	fl::RuleBlock* removeRuleBlock(const std::string& name);

	template <typename IterT>
	void setRuleBlocks(IterT first, IterT last)
	{
		ruleBlocks_.clear();
		ruleBlocks_.assign(first, last);
	}

	void setRuleBlocks(const std::vector<fl::RuleBlock*>& ruleBlocks);

	std::vector<fl::RuleBlock*> const& ruleBlocks() const;

	//std::size_t numberOfRuleBlocks() const;
	int numberOfRuleBlocks() const;

	void setInputValue(const std::string& name, fl::scalar value);

	fl::scalar getOutputValue(const std::string& name) const;

	bool isReady(std::string* status = fl::null) const;

	void process();

	void restart();

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
	void setInputs(IterT first, IterT last)
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
	std::vector<fl::scalar> eval(IterT first, IterT last)
	{
		this->setInputs(first, last);
		return this->eval();
	}

	template <typename IterT>
	std::vector<fl::scalar> evalTo(IterT first, IterT last, LayerCategory layer)
	{
		this->setInputs(first, last);
		return this->evalTo(layer);
	}

	LayerCategory getNextLayerCategory(LayerCategory cat) const;

	LayerCategory getPreviousLayerCategory(LayerCategory cat) const;

	void clear();

private:
	void check();

	void connect(Node* p_from, Node* p_to);

	std::vector<fl::scalar> evalLayer(LayerCategory layer);

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

	std::vector<fl::scalar> evalInputHedgeLayer();

	std::vector<fl::scalar> evalAntecedentLayer();

	std::vector<fl::scalar> evalConsequentLayer();

	std::vector<fl::scalar> evalAccumulationLayer();

	std::vector<fl::scalar> evalOutputLayer();

private:
	std::string name_; ///< The mnemonic name for this FIS engine
	std::vector<fl::InputVariable*> inputs_; ///< Collection of (pointer to) input variables
	std::vector<fl::OutputVariable*> outputs_; ///< Collection of (pointer to) output variables
	std::vector<fl::RuleBlock*> ruleBlocks_; ///< Collection of (pointer to) rule blocks
	//std::size_t order_;
	std::vector<InputNode*> inputNodes_; ///< Nodes in the input layer
	std::vector<FuzzificationNode*> fuzzificationNodes_; ///< Nodes in the fuzzification layer for linguistic terms
	std::vector<InputHedgeNode*> inputHedgeNodes_; ///< Additional nodes in the fuzzification layer for linguistic hedges
	std::vector<AntecedentNode*> antecedentNodes_; ///< Nodes in the antecedent layer
	std::vector<ConsequentNode*> consequentNodes_; ///< Nodes in the consequent layer
	std::vector<AccumulationNode*> accumulationNodes_; ///< Nodes in the summation layer
	std::vector<OutputNode*> outputNodes_; ///< Nodes in the inference layer
	std::map< const Node*, std::vector<Node*> > inConns_; ///< Input connection to a given node
	std::map< const Node*, std::vector<Node*> > outConns_; ///< Output connection from a given node
}; // Engine

}} // Namespace fl::anfis

#endif // FL_ANFIS_ENGINE_H
