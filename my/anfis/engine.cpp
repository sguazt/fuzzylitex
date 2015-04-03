#include <cstddef>
#include <fl/anfis/engine.h>
#include <fl/commons.h>
#include <fl/detail/derivatives.h>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/Headers.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>


namespace fl { namespace anfis {

namespace detail {

namespace /*<unnamed>*/ {

void flattenRuleAntecedentRec(fl::Expression* p_expr, std::vector<fl::Variable*>& vars, std::vector<fl::Term*>& terms, std::vector<bool>& nots, std::string& opKeyword)
{
	if (dynamic_cast<fl::Proposition*>(p_expr))
	{
		// The expression is a single simple statement like "X is FOO"

		const fl::Proposition* p_prop = dynamic_cast<fl::Proposition*>(p_expr);

		const std::string notKeyword = fl::Not().name();

		bool foundNot = false;
		for (std::size_t h = 0,
						 nh = p_prop->hedges.size();
			 h < nh && !foundNot;
			 ++h)
		{
			if (p_prop->hedges[h]->name() == notKeyword)
			{
				foundNot = true;
			}
		}

		nots.push_back(foundNot);
		terms.push_back(p_prop->term);
		vars.push_back(p_prop->variable);
	}
	else if (dynamic_cast<fl::Operator*>(p_expr))
	{
		// The expression is a compound statement like "X is FOO and Y is BAR"

		fl::Operator* p_op = dynamic_cast<fl::Operator*>(p_expr);

		if (opKeyword.empty())
		{
			opKeyword = p_op->name;
		}
		else if (opKeyword != p_op->name)
		{
			FL_THROW2(std::runtime_error, "Rules with mixed AND/OR operators are not yet supported by ANFIS");
		}

		flattenRuleAntecedentRec(p_op->left, vars, terms, nots, opKeyword);
		flattenRuleAntecedentRec(p_op->right, vars, terms, nots, opKeyword);
	}
}

void flattenRuleAntecedent(fl::Antecedent* p_antecedent, std::vector<fl::Variable*>& vars, std::vector<fl::Term*>& terms, std::vector<bool>& nots, std::string& opKeyword)
{
	flattenRuleAntecedentRec(p_antecedent->getExpression(), vars, terms, nots, opKeyword);

	if (!terms.empty() && opKeyword.empty())
	{
		opKeyword = fl::Rule::andKeyword();
	}
}

} // Namespace <unnamed>

} // Namespace detail

////////////////////////////////////////////////////////////////////////////////
// Nodes
////////////////////////////////////////////////////////////////////////////////

/////////////
// Node
/////////////


Node::Node(Engine* p_engine)
: p_engine_(p_engine)
{
}

Node::~Node()
{
	// empty
}

void Node::setEngine(Engine* p_engine)
{
	p_engine_ = p_engine;
}

Engine* Node::getEngine() const
{
	return p_engine_;
}

std::vector<Node*> Node::inputConnections() const
{
	return p_engine_->inputConnections(this);
}

std::vector<Node*> Node::outputConnections() const
{
	return p_engine_->outputConnections(this);
}

std::vector<fl::scalar> Node::inputs() const
{
	std::vector<fl::scalar> inps;

	std::vector<Node*> inConns = p_engine_->inputConnections(this);
	for (typename std::vector<Node*>::const_iterator it = inConns.begin(),
													 endIt = inConns.end();
		 it != endIt;
		 ++it)
	{
		const Node* p_node = *it;

		FL_DEBUG_ASSERT( p_node );

		inps.push_back(p_node->getValue());
	}

	return inps;
}

fl::scalar Node::eval()
{
	val_ = this->doEval();

	return val_;
}

std::vector<fl::scalar> Node::evalDerivativeWrtInputs()
{
	return this->doEvalDerivativeWrtInputs();
}

fl::scalar Node::getValue() const
{
	return val_;
}

void Node::setValue(fl::scalar v)
{
	val_ = v;
}

/////////////
// Input Node
/////////////

InputNode::InputNode(fl::InputVariable* p_var, Engine* p_engine)
: Node(p_engine),
  p_var_(p_var)
{
}

fl::InputVariable* InputNode::getInputVariable() const
{
	return p_var_;
}

fl::scalar InputNode::doEval()
{
	return p_var_->getInputValue();
}

std::vector<fl::scalar> InputNode::doEvalDerivativeWrtInputs()
{
	FL_THROW2(std::logic_error, "Derivative wrt inputs should not be evaluated for input nodes ");
}

/////////////
// Fuzzification Node
/////////////

FuzzificationNode::FuzzificationNode(fl::Term* p_term, Engine* p_engine)
: Node(p_engine),
  p_term_(p_term)
{
}

fl::Term* FuzzificationNode::getTerm() const
{
	return p_term_;
}

std::vector<fl::scalar> FuzzificationNode::evalDerivativeWrtParams()
{
	const std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Fuzzification node must have exactly one input");
	}

	return fl::detail::EvalTermDerivativeWrtParams(p_term_, inputs.back());
}

fl::scalar FuzzificationNode::doEval()
{
	const std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Fuzzification node must have exactly one input");
	}

	return p_term_->membership(inputs.front());
}

std::vector<fl::scalar> FuzzificationNode::doEvalDerivativeWrtInputs()
{
	FL_THROW2(std::logic_error, "Derivative wrt inputs should not be evaluated for fuzzification nodes ");
}

/////////////
// Input Hedge Node
/////////////

InputHedgeNode::InputHedgeNode(fl::Hedge* p_hedge, Engine* p_engine)
: Node(p_engine),
  p_hedge_(p_hedge)
{
}

InputHedgeNode::~InputHedgeNode()
{
	if (p_hedge_)
	{
		delete p_hedge_;
	}
}

fl::Hedge* InputHedgeNode::getHedge() const
{
	return p_hedge_;
}

fl::scalar InputHedgeNode::doEval()
{
	const std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Hedge node must have exactly one input");
	}

	return p_hedge_->hedge(inputs.front());
}

std::vector<fl::scalar> InputHedgeNode::doEvalDerivativeWrtInputs()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Hedge node must have exactly one input");
	}

	return std::vector<fl::scalar>(inputs.size(), -1);
}

/////////////
// Antecedent Node
/////////////

AntecedentNode::AntecedentNode(fl::Norm* p_norm, Engine* p_engine)
: Node(p_engine),
  p_norm_(p_norm)
{
}

fl::Norm* AntecedentNode::getNorm() const
{
	return p_norm_;
}

fl::scalar AntecedentNode::doEval()
{
	const std::vector<fl::scalar> inputs = this->inputs();

	fl::scalar res = fl::nan;

	if (inputs.size() > 0)
	{
		res = inputs.front();

		for (std::size_t i = 1,
						 n = inputs.size();
			 i < n;
			 ++i)
		{
			res = p_norm_->compute(res, inputs[i]);
		}
	}

	return res;
}

std::vector<fl::scalar> AntecedentNode::doEvalDerivativeWrtInputs()
{
	std::vector<fl::scalar> res;

	const std::vector<fl::scalar> inputs = this->inputs();

	if (dynamic_cast<fl::Minimum*>(p_norm_)
		|| dynamic_cast<fl::Maximum*>(p_norm_))
	{
		// The derivative is 0 for all inputs, excepts for the one
		// corresponding to the min (or max) for which the derivative is 1

		const fl::scalar fx = this->eval();

		for (std::size_t i = 0,
						 n = inputs.size();
			 i < n;
			 ++i)
		{
			const fl::scalar in = inputs[i];

			if (fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(fx, in))
			{
				res.push_back(1);
			}
			else
			{
				res.push_back(0);
			}
		}
	}
	else if (dynamic_cast<fl::AlgebraicProduct*>(p_norm_))
	{
		// The derivative for a certain input i is given by the product of
		// all the inputs but i

		for (std::size_t i = 0,
						 n = inputs.size();
			 i < n;
			 ++i)
		{
			fl::scalar prod = 1;
			for (std::size_t j = 0; j < n; ++j)
			{
				if (j != i)
				{
					const fl::scalar in = inputs[j];

					prod *= in;
				}
			}

			res.push_back(prod);
		}
	}
	else if (dynamic_cast<fl::AlgebraicSum*>(p_norm_))
	{
		// The derivative of each input is given by:
		//  \frac{\partial [x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3 + x_4 - (x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3)*x_4 + ...]}{\partial x_1}
		//  = 1 - x_2 - x_3 - x_2*x_3 - x_4 - x_2*x_4 - x_3*x_4 - x_2*x_3*x_4 + ...
		//  = (1-x2)*(1-x3)*(1-x4)*...

		for (std::size_t i = 0,
						 n = inputs.size();
			 i < n;
			 ++i)
		{
			fl::scalar prod = 1;
			for (std::size_t j = 0; j < n; ++j)
			{
				if (j != i)
				{
					const fl::scalar in = inputs[i];

					prod *= (1-in);
				}
			}

			res.push_back(prod);
		}
	}
	else
	{
		FL_THROW2(std::runtime_error, "Norm operator '" + p_norm_->className() + "' not yet implemented");
	}

	return res;
}

/////////////
// Consequent Node
/////////////

ConsequentNode::ConsequentNode(fl::Term* p_term, fl::TNorm* p_tnorm, Engine* p_engine)
: Node(p_engine),
  p_term_(p_term),
  p_tnorm_(p_tnorm)
{
}

fl::Term* ConsequentNode::getTerm() const
{
	return p_term_;
}

fl::TNorm* ConsequentNode::getTNorm() const
{
	return p_tnorm_;
}

fl::scalar ConsequentNode::doEval()
{
	const std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Consequent node must have exactly one input");
	}

	// The last and only input is the one coming from the antecedent layer.
	return p_tnorm_->compute(inputs.back(), p_term_->membership(1.0));
}

std::vector<fl::scalar> ConsequentNode::doEvalDerivativeWrtInputs()
{
//{//[XXX]
//std::cerr << "PHASE #1 - Backward pass for Consequent Node: ";
//if (dynamic_cast<fl::Linear*>(p_term_))
//{
//	fl::Linear* p_linear = dynamic_cast<fl::Linear*>(p_term_);
//	for (std::size_t i = 0; i < p_linear->getEngine()->inputVariables().size(); ++i) {
//		if (i < p_linear->coefficients().size())
//		{
//			std::cerr << " + " << p_linear->getEngine()->inputVariables().at(i)->getInputValue() << "*" << p_linear->coefficients().at(i);
//		}
//	}
//	if (p_linear->coefficients().size() > p_linear->getEngine()->inputVariables().size())
//	{
//		std::cerr << "+ " << p_linear->coefficients().back();
//	}
//}
//else
//{
//	fl::Constant* p_const = dynamic_cast<fl::Constant*>(p_term_);
//	std::cerr << p_const->getValue();
//}
//std::cerr << std::endl;
//}//[/XXX]
	return std::vector<fl::scalar>(1, p_term_->membership(1.0));
}

/////////////
// Accumulation Node
/////////////

AccumulationNode::AccumulationNode(Engine* p_engine)
: Node(p_engine)
{
}

fl::scalar AccumulationNode::doEval()
{
	const std::vector<fl::scalar> inputs = this->inputs();

	fl::scalar sum = 0;

	for (std::size_t i = 0,
					 n = inputs.size();
		 i < n;
		 ++i)
	{
		sum += inputs[i];
	}

	return sum;
}

std::vector<fl::scalar> AccumulationNode::doEvalDerivativeWrtInputs()
{
	return std::vector<fl::scalar>(this->inputs().size(), 1.0);
}

/////////////
// Output Node
/////////////

OutputNode::OutputNode(Engine* p_engine)
: Node(p_engine)
{
}

fl::scalar OutputNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 2)
	{
		FL_THROW2(std::logic_error, "Output node must have exactly two inputs");
	}

	if (fl::detail::FloatTraits<fl::scalar>::ApproximatelyZero(inputs[1]))
	{
		return fl::nan;
	}

	return inputs[0]/inputs[1];
}

std::vector<fl::scalar> OutputNode::doEvalDerivativeWrtInputs()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 2)
	{
		FL_THROW2(std::logic_error, "Output node must have exactly two inputs");
	}

	if (fl::detail::FloatTraits<fl::scalar>::ApproximatelyZero(inputs[1]))
	{
		return std::vector<fl::scalar>(inputs.size(), fl::nan);
	}

	std::vector<fl::scalar> res(2);
	res[0] = 1.0/inputs[1];
	res[1] = -inputs[0]/fl::detail::Sqr(inputs[1]);

	return res;
}


/////////
// Engine
/////////

Engine::Engine(const std::string& name)
: name_(name)
{
}

//TODO
Engine::Engine(const Engine& other)
{
	throw std::runtime_error("Engine Copy ctor to be implemented");
}

Engine::~Engine()
{
	this->clear();
}

//TODO
Engine& Engine::operator=(const Engine& rhs)
{
	throw std::runtime_error("Engine::operator= to be implemented");
	return *this;
}

//TODO
Engine* Engine::clone() const
{
	throw std::runtime_error("Engine::clone to be implemented");
}

//TODO
std::string Engine::toString() const
{
	return FllExporter().toString(this);
}

//TODO
void Engine::configure(fl::TNorm* conjunction, fl::SNorm* disjunction,
					   fl::TNorm* activation, fl::SNorm* accumulation,
					   fl::Defuzzifier* defuzzifier)
{
	throw std::runtime_error("Engine::configure to be implemented");
} 

//TODO
void Engine::configure(const std::string& conjunctionT,
					   const std::string& disjunctionS,
					   const std::string& activationT,
					   const std::string& accumulationS,
					   const std::string& defuzzifier,
					   int resolution)
{
	throw std::runtime_error("Engine::configure to be implemented");
}

void Engine::setName(const std::string& name)
{
	name_ = name;
}

std::string Engine::getName() const
{
	return name_;
}

void Engine::addInputVariable(fl::InputVariable* p_var)
{
	if (!p_var)
	{
		FL_THROW2(std::invalid_argument, "Input variable cannot be null");
	}

	inputs_.push_back(p_var);
}

//fl::InputVariable* setInputVariable(fl::InputVariable* p_var, std::size_t idx);
fl::InputVariable* Engine::setInputVariable(fl::InputVariable* p_var, int idx)
{
	if (idx >= inputs_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to input variable is out of range");
	}

	fl::InputVariable* p_oldVar = inputs_[idx];

	inputs_[idx] = p_var;

	return p_oldVar;
}

//void insertInputVariable(fl::InputVariable* p_var, std::size_t idx);
void Engine::insertInputVariable(fl::InputVariable* p_var, int idx)
{
	if (!p_var)
	{
		FL_THROW2(std::invalid_argument, "Input variable cannot be null")
	}

	inputs_.insert(inputs_.begin()+idx, p_var);
}

//fl::InputVariable* Engine::getInputVariable(std::size_t idx) const
fl::InputVariable* Engine::getInputVariable(int idx) const
{
	if (idx >= inputs_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to input variable is out of range");
	}

	return inputs_[idx];
}

fl::InputVariable* Engine::getInputVariable(const std::string& name) const
{
	for (std::size_t i = 0,
					 n = inputs_.size();
		 i < n;
		 ++i)
	{
		fl::InputVariable* p_input = inputs_[i];

		//check: null
		FL_DEBUG_ASSERT( p_input );

		if (p_input->getName() == name)
		{
			return p_input;
		}
	}

	FL_THROW("Input variable <" + name + "> not found");
}

//fl::InputVariable* Engine::removeInputVariable(std::size_t idx)
fl::InputVariable* Engine::removeInputVariable(int idx)
{
	if (idx >= inputs_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to input variable is out of range");
	}

	fl::InputVariable* p_var = inputs_[idx];

	inputs_.erase(inputs_.begin()+idx);

	return p_var;
}

fl::InputVariable* Engine::removeInputVariable(const std::string& name)
{
	for (typename std::vector<fl::InputVariable*>::iterator it = inputs_.begin(),
															endIt = inputs_.end();
		 it != endIt;
		 ++it)
	{
		fl::InputVariable* p_var = *it;

		if (p_var->getName() == name)
		{
			inputs_.erase(it);
			return p_var;
		}
	}

	FL_THROW("Input variable <" + name + "> not found");
}

bool Engine::hasInputVariable(const std::string& name) const
{
	for (std::size_t i =0,
					 n = inputs_.size();
		 i < n;
		 ++i)
	{
		fl::InputVariable* p_var = inputs_[i];

		if (p_var->getName() == name)
		{
			return true;
		}
	}

	return false;
}

void Engine::setInputVariables(const std::vector<fl::InputVariable*>& vars)
{
	this->setInputVariables(vars.begin(), vars.end());
}

std::vector<fl::InputVariable*> const& Engine::inputVariables() const
{
	return inputs_;
}

//std::size_t Engine::numberOfInputVariables() const
int Engine::numberOfInputVariables() const
{
	return inputs_.size();
}

void Engine::addOutputVariable(fl::OutputVariable* p_var)
{
	if (!p_var)
	{
		FL_THROW2(std::invalid_argument, "Output variable cannot be null");
	}

	outputs_.push_back(p_var);
}

//fl::OutputVariable* setOutputVariable(fl::OutputVariable* p_var, std::size_t idx);
fl::OutputVariable* Engine::setOutputVariable(fl::OutputVariable* p_var, int idx)
{
	if (idx >= outputs_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to output variable is out of range");
	}

	fl::OutputVariable* p_oldVar = outputs_[idx];

	outputs_[idx] = p_var;

	return p_oldVar;
}

//void insertOutputVariable(fl::OutputVariable* p_var, std::size_t idx);
void Engine::insertOutputVariable(fl::OutputVariable* p_var, int idx)
{
	if (!p_var)
	{
		FL_THROW2(std::invalid_argument, "Output variable cannot be null")
	}

	outputs_.insert(outputs_.begin()+idx, p_var);
}

//fl::OutputVariable* Engine::getOutputVariable(std::size_t idx) const
fl::OutputVariable* Engine::getOutputVariable(int idx) const
{
	if (idx >= outputs_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to output variable is out of range");
	}

	return outputs_[idx];
}

fl::OutputVariable* Engine::getOutputVariable(const std::string& name) const
{
	for (std::size_t i = 0,
					 n = outputs_.size();
		 i < n;
		 ++i)
	{
		fl::OutputVariable* p_output = outputs_[i];

		//check: null
		FL_DEBUG_ASSERT( p_output );

		if (p_output->getName() == name)
		{
			return p_output;
		}
	}

	FL_THROW("Output variable <" + name + "> not found");
}

//fl::OutputVariable* Engine::removeOutputVariable(std::size_t idx)
fl::OutputVariable* Engine::removeOutputVariable(int idx)
{
	if (idx >= outputs_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to output variable is out of range");
	}

	fl::OutputVariable* p_var = outputs_[idx];

	outputs_.erase(outputs_.begin()+idx);

	return p_var;
}

fl::OutputVariable* Engine::removeOutputVariable(const std::string& name)
{
	for (typename std::vector<fl::OutputVariable*>::iterator it = outputs_.begin(),
															 endIt = outputs_.end();
		 it != endIt;
		 ++it)
	{
		fl::OutputVariable* p_var = *it;

		if (p_var->getName() == name)
		{
			outputs_.erase(it);
			return p_var;
		}
	}

	FL_THROW("Output variable <" + name + "> not found");
}

bool Engine::hasOutputVariable(const std::string& name) const
{
	for (std::size_t i =0,
					 n = outputs_.size();
		 i < n;
		 ++i)
	{
		fl::OutputVariable* p_var = outputs_[i];

		if (p_var->getName() == name)
		{
			return true;
		}
	}

	return false;
}

void Engine::setOutputVariables(const std::vector<fl::OutputVariable*>& vars)
{
	this->setOutputVariables(vars.begin(), vars.end());
}

std::vector<fl::OutputVariable*> const& Engine::outputVariables() const
{
	return outputs_;
}

//std::size_t Engine::numberOfOutputVariables() const
int Engine::numberOfOutputVariables() const
{
	return outputs_.size();
}

std::vector<fl::Variable*> Engine::variables() const
{
	std::vector<fl::Variable*> vars;

	vars.insert(vars.begin(), inputs_.begin(), inputs_.end());
	vars.insert(vars.end(), outputs_.begin(), outputs_.end());

	return vars;
}

void Engine::addRuleBlock(fl::RuleBlock* p_block)
{
	//FIXME: what does it happens if multiple rule blocks are enabled and share the same output variables?

	if (!p_block)
	{
		FL_THROW2(std::invalid_argument, "Rule block cannot be null");
	}

	ruleBlocks_.push_back(p_block);
}

//fl::RuleBlock* Engine::setRuleBlock(fl::RuleBlock* p_block, std::size_t idx)
fl::RuleBlock* Engine::setRuleBlock(fl::RuleBlock* p_block, int idx)
{
	if (idx >= ruleBlocks_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to rule block is out of range");
	}

	fl::RuleBlock* p_oldBlock = ruleBlocks_[idx];

	ruleBlocks_.erase(ruleBlocks_.begin()+idx);

	return p_oldBlock;
}

//void Engine::insertRuleBlock(fl::RuleBlock* p_block, std::size_t idx)
void Engine::insertRuleBlock(fl::RuleBlock* p_block, int idx)
{
	if (!p_block)
	{
		FL_THROW2(std::invalid_argument, "Rule block cannot be null")
	}


	ruleBlocks_.insert(ruleBlocks_.begin()+idx, p_block);
}

//fl::RuleBlock* Engine::getRuleBlock(std::size_t idx) const
fl::RuleBlock* Engine::getRuleBlock(int idx) const
{
	if (idx >= ruleBlocks_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to rule block is out of range");
	}

	return ruleBlocks_[idx];
}

fl::RuleBlock* Engine::getRuleBlock(const std::string& name) const
{
	for (std::size_t i = 0,
					 n = ruleBlocks_.size();
		 i < n;
		 ++i)
	{
		fl::RuleBlock* p_block = ruleBlocks_[i];

		//check: null
		FL_DEBUG_ASSERT( p_block );

		if (p_block->getName() == name)
		{
			return p_block;
		}
	}

	FL_THROW("Rule block <" + name + "> not found");
}

//fl::RuleBlock* Engine::removeRuleBlock(std::size_t idx)
fl::RuleBlock* Engine::removeRuleBlock(int idx)
{
	if (idx >= ruleBlocks_.size())
	{
		FL_THROW2(std::invalid_argument, "Index to rule block is out of range");
	}

	fl::RuleBlock* p_block = ruleBlocks_[idx];

	ruleBlocks_.erase(ruleBlocks_.begin()+idx);

	return p_block;
}

fl::RuleBlock* Engine::removeRuleBlock(const std::string& name)
{
	for (typename std::vector<fl::RuleBlock*>::iterator it = ruleBlocks_.begin(),
															 endIt = ruleBlocks_.end();
		 it != endIt;
		 ++it)
	{
		fl::RuleBlock* p_block = *it;

		if (p_block->getName() == name)
		{
			ruleBlocks_.erase(it);
			return p_block;
		}
	}

	FL_THROW("Rule block <" + name + "> not found");
}

bool Engine::hasRuleBlock(const std::string& name) const
{
	for (std::size_t i =0,
					 n = ruleBlocks_.size();
		 i < n;
		 ++i)
	{
		fl::RuleBlock* p_block = ruleBlocks_[i];

		if (p_block->getName() == name)
		{
			return true;
		}
	}

	return false;
}

void Engine::setRuleBlocks(const std::vector<fl::RuleBlock*>& ruleBlocks)
{
	this->setRuleBlocks(ruleBlocks.begin(), ruleBlocks.end());
}

std::vector<fl::RuleBlock*> const& Engine::ruleBlocks() const
{
	return ruleBlocks_;
}

//std::size_t Engine::numberOfRuleBlocks() const
int Engine::numberOfRuleBlocks() const
{
	return ruleBlocks_.size();
}

void Engine::setInputValue(const std::string& name, fl::scalar value)
{
	fl::InputVariable* p_iv = this->getInputVariable(name);

	p_iv->setInputValue(value);
}

fl::scalar Engine::getOutputValue(const std::string& name) const
{
	const fl::OutputVariable* p_ov = this->getOutputVariable(name);

	return p_ov->getOutputValue();
}

//TODO
bool Engine::isReady(std::string* status) const
{
	throw std::runtime_error("Engine::isReady to be implemented");
}

//TODO
void Engine::process()
{
	throw std::runtime_error("Engine::process to be implemented");
}

//TODO
void Engine::restart()
{
	throw std::runtime_error("Engine::restart to be implemented");
}

std::vector<fl::scalar> Engine::getInputValues() const
{
	const std::size_t n = inputNodes_.size();

	std::vector<fl::scalar> inputs(n);

	for (std::size_t i = 0; i < n; ++i)
	{
		inputs[i] = inputNodes_[i]->getInputVariable()->getInputValue();
	}

	return inputs;
}

std::vector<InputNode*> Engine::getInputLayer() const
{
	return inputNodes_;
}

std::vector<FuzzificationNode*> Engine::getFuzzificationLayer() const
{
	return fuzzificationNodes_;
}

std::vector<InputHedgeNode*> Engine::getInputHedgeLayer() const
{
	return inputHedgeNodes_;
}

std::vector<AntecedentNode*> Engine::getAntecedentLayer() const
{
	return antecedentNodes_;
}

std::vector<ConsequentNode*> Engine::getConsequentLayer() const
{
	return consequentNodes_;
}

std::vector<AccumulationNode*> Engine::getAccumulationLayer() const
{
	return accumulationNodes_;
}

std::vector<OutputNode*> Engine::getOutputLayer() const
{
	return outputNodes_;
}

std::vector<Node*> Engine::getLayer(Engine::LayerCategory layerCat) const
{
	switch (layerCat)
	{
		case Engine::InputLayer:
			return std::vector<Node*>(inputNodes_.begin(), inputNodes_.end());
		case Engine::FuzzificationLayer:
			return std::vector<Node*>(fuzzificationNodes_.begin(), fuzzificationNodes_.end());
		case Engine::InputHedgeLayer:
			return std::vector<Node*>(inputHedgeNodes_.begin(), inputHedgeNodes_.end());
		case Engine::AntecedentLayer:
			return std::vector<Node*>(antecedentNodes_.begin(), antecedentNodes_.end());
		case Engine::ConsequentLayer:
			return std::vector<Node*>(consequentNodes_.begin(), consequentNodes_.end());
		case Engine::AccumulationLayer:
			return std::vector<Node*>(accumulationNodes_.begin(), accumulationNodes_.end());
		case Engine::OutputLayer:
			return std::vector<Node*>(outputNodes_.begin(), outputNodes_.end());
	}

	FL_THROW2(std::runtime_error, "Unexpected error");
}

std::vector<fl::scalar> Engine::evalInputLayer()
{
	return this->evalLayer(inputNodes_.begin(), inputNodes_.end());
}

std::vector<fl::scalar> Engine::evalFuzzificationLayer()
{
	return this->evalLayer(fuzzificationNodes_.begin(), fuzzificationNodes_.end());
}

std::vector<fl::scalar> Engine::evalInputHedgeLayer()
{
	return this->evalLayer(inputHedgeNodes_.begin(), inputHedgeNodes_.end());
}

std::vector<fl::scalar> Engine::evalAntecedentLayer()
{
	return this->evalLayer(antecedentNodes_.begin(), antecedentNodes_.end());
}

std::vector<fl::scalar> Engine::evalConsequentLayer()
{
	return this->evalLayer(consequentNodes_.begin(), consequentNodes_.end());
}

std::vector<fl::scalar> Engine::evalAccumulationLayer()
{
	return this->evalLayer(accumulationNodes_.begin(), accumulationNodes_.end());
}

std::vector<fl::scalar> Engine::evalOutputLayer()
{
	return this->evalLayer(outputNodes_.begin(), outputNodes_.end());
}

std::vector<fl::scalar> Engine::eval()
{
/*
std::cerr << "Evaluating ANFIS against input: "; fl::detail::VectorOutput(std::cerr, this->getInputValues()); std::cerr << std::endl;//XXX
	std::vector<fl::scalar> out;
	// Eval input layer
	out = this->evalInputLayer();
std::cerr << "- Output from Layer " << Engine::InputLayer << ": "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl;//XXX
	// Eval fuzzification layer
	out = this->evalFuzzificationLayer();
std::cerr << "- Output from Layer " << Engine::FuzzificationLayer << ": "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl;//XXX
	// Eval hedge layer
	out = this->evalInputHedgeLayer();
std::cerr << "- Output from Layer " << Engine::InputHedgeLayer << ": "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl;//XXX
	// Eval rule antecedent layer
	out = this->evalAntecedentLayer();
std::cerr << "- Output from Layer " << Engine::AntecedentLayer << ": "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl;//XXX
	// Eval rule consequent layer
	out = this->evalConsequentLayer();
std::cerr << "- Output from Layer " << Engine::ConsequentLayer << ": "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl;//XXX
	// Eval rule accumulation layer
	out = this->evalAccumulationLayer();
std::cerr << "- Output from Layer " << Engine::AccumulationLayer << ": "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl;//XXX
	// Eval rule strength normalization layer
	out = this->evalOutputLayer();
std::cerr << "- Output from Layer " << Engine::OutputLayer << ": "; fl::detail::VectorOutput(std::cerr, out); std::cerr << std::endl;//XXX
	return out;
*/
	// Eval input layer
	this->evalInputLayer();
	// Eval fuzzification layer
	this->evalFuzzificationLayer();
	// Eval hedge layer
	this->evalInputHedgeLayer();
	// Eval rule antecedent layer
	this->evalAntecedentLayer();
	// Eval rule consequent layer
	this->evalConsequentLayer();
	// Eval rule accumulation layer
	this->evalAccumulationLayer();
	// Eval rule strength normalization layer
	return this->evalOutputLayer();
}

std::vector<fl::scalar> Engine::evalTo(Engine::LayerCategory layer)
{
	std::vector<fl::scalar> res;

	// Eval input layer
	res = this->evalInputLayer();
	if (layer == Engine::InputLayer)
	{	
		return res;
	}
	// Eval fuzzification layer
	res = this->evalFuzzificationLayer();
	if (layer == Engine::FuzzificationLayer)
	{
		return res;
	}
	// Eval hedge layer
	res = this->evalInputHedgeLayer();
	if (layer == Engine::InputHedgeLayer)
	{
		return res;
	}
	// Eval rule antecedent layer
	res = this->evalAntecedentLayer();
	if (layer == Engine::AntecedentLayer)
	{
		return res;
	}
	// Eval rule consequent layer
	res = this->evalConsequentLayer();
	if (layer == Engine::ConsequentLayer)
	{
		return res;
	}
	// Eval rule accumulation layer
	res = this->evalAccumulationLayer();
	if (layer == Engine::AccumulationLayer)
	{
		return res;
	}
	// Eval rule strength normalization layer
	return this->evalOutputLayer();
}

std::vector<fl::scalar> Engine::evalFrom(Engine::LayerCategory layer)
{
	while (layer < Engine::OutputLayer)
	{
		this->evalLayer(layer);
		//layer = static_cast<int>(layer)+1;
		layer = static_cast<Engine::LayerCategory>(layer+1);
	}
	return this->evalOutputLayer();
}

std::vector<fl::scalar> Engine::evalLayer(Engine::LayerCategory layer)
{
	switch (layer)
	{
		case Engine::InputLayer:
			return this->evalInputLayer();
		case Engine::FuzzificationLayer:
			return this->evalFuzzificationLayer();
		case Engine::InputHedgeLayer:
			return this->evalInputHedgeLayer();
		case Engine::AntecedentLayer:
			return this->evalAntecedentLayer();
		case Engine::ConsequentLayer:
			return this->evalConsequentLayer();
		case Engine::AccumulationLayer:
			return this->evalAccumulationLayer();
		case Engine::OutputLayer:
			return this->evalOutputLayer();
	}

	FL_THROW2(std::runtime_error, "Unexpected error");
}

Engine::LayerCategory Engine::getNextLayerCategory(Engine::LayerCategory cat) const
{
	if (cat == Engine::OutputLayer)
	{
		return Engine::OutputLayer;
	}

	int nextCat = static_cast<int>(cat)+1;

	return static_cast<Engine::LayerCategory>(nextCat);
}

Engine::LayerCategory Engine::getPreviousLayerCategory(Engine::LayerCategory cat) const
{
	if (cat == Engine::InputLayer)
	{
		return Engine::InputLayer;
	}

	int prevCat = static_cast<int>(cat)-1;

	return static_cast<Engine::LayerCategory>(prevCat);
}

void Engine::clear()
{
	inConns_.clear();
	outConns_.clear();

	for (std::size_t i = 0,
					 n = inputs_.size();
		 i < n;
		 ++i)
	{
		delete inputs_[i];
	}
	inputs_.clear();

	for (std::size_t i = 0,
					 n = outputs_.size();
		 i < n;
		 ++i)
	{
		delete outputs_[i];
	}
	outputs_.clear();

	for (std::size_t i = 0,
					 n = ruleBlocks_.size();
		 i < n;
		 ++i)
	{
		delete ruleBlocks_[i];
	}
	ruleBlocks_.clear();

	for (std::size_t i = 0,
					 n = inputNodes_.size();
		 i < n;
		 ++i)
	{
		delete inputNodes_[i];
	}
	inputNodes_.clear();

	for (std::size_t i = 0,
					 n = fuzzificationNodes_.size();
		 i < n;
		 ++i)
	{
		delete fuzzificationNodes_[i];
	}
	fuzzificationNodes_.clear();

	for (std::size_t i = 0,
					 n = inputHedgeNodes_.size();
		 i < n;
		 ++i)
	{
		delete inputHedgeNodes_[i];
	}
	inputHedgeNodes_.clear();

	for (std::size_t i = 0,
					 n = antecedentNodes_.size();
		 i < n;
		 ++i)
	{
		delete antecedentNodes_[i];
	}
	antecedentNodes_.clear();

	for (std::size_t i = 0,
					 n = consequentNodes_.size();
		 i < n;
		 ++i)
	{
		delete consequentNodes_[i];
	}
	consequentNodes_.clear();

	for (std::size_t i = 0,
					 n = accumulationNodes_.size();
		 i < n;
		 ++i)
	{
		delete accumulationNodes_[i];
	}
	accumulationNodes_.clear();

	for (std::size_t i = 0,
					 n = outputNodes_.size();
		 i < n;
		 ++i)
	{
		delete outputNodes_[i];
	}
	outputNodes_.clear();
}

std::vector<Node*> Engine::inputConnections(const Node* p_node) const
{
	return inConns_.count(p_node) > 0 ? inConns_.at(p_node) : std::vector<Node*>();
}

std::vector<Node*> Engine::outputConnections(const Node* p_node) const
{
	return outConns_.count(p_node) > 0 ? outConns_.at(p_node) : std::vector<Node*>();
}

void Engine::check()
{
	// Check output var
	if (outputs_.size() != 1)
	{
		FL_THROW2(std::logic_error, "There must be exactly one output variable");
	}

	//NOTE: order is valid only for Takagi-Sugeno ANFIS
	//		// Check order
	//		std::size_t order = 0;
	//		for (std::size_t t = 0,
	//						 nt = p_output_->numberOfTerms();
	//			 t != nt;
	//			 ++t)
	//		{
	//			const fl::Term* p_term = p_output->getTerm(t);
	//
	//			// check: null
	//			FL_DEBUG_ASSERT( p_term );
	//
	//			if (dynamic_cast<fl::Constant const*>(p_term))
	//			{
	//				if (t > 0 && order != 0)
	//				{
	//					FL_THROW2(std::logic_error, "Output terms must be of the same order");
	//				}
	//
	//				order = 0;
	//			}
	//			else if (dynamic_cast<fl::Linear const*>(p_term))
	//			{
	//				if (t > 0 && order != 1)
	//				{
	//					FL_THROW2(std::logic_error, "Output terms must be of the same order");
	//				}
	//
	//				order = 1;
	//			}
	//		}
}

void Engine::build()
{
	this->check();

	std::map<const fl::Variable*,Node*> varNodeMap;
	std::map<const fl::Term*,Node*> termNodeMap;
	std::map<const fl::Term*,Node*> notFuzzificationNodeMap;
	std::map<const fl::Rule*,Node*> ruleAntecedentNodeMap;

	// Layer 0 (the input layer): input linguistic variables
	// There is one node for each input variable
	for (std::size_t i = 0,
					 n = inputs_.size();
		 i < n;
		 ++i)
	{
		fl::InputVariable* p_input = inputs_[i];

		// check: null
		FL_DEBUG_ASSERT( p_input );

		InputNode* p_node = new InputNode(p_input, this);
		inputNodes_.push_back(p_node);

		varNodeMap[p_input] = p_node;
	}

	// Layer 1: linguistic terms layer
	// There is one node for each linguistic term of each input variable
	for (std::size_t i = 0,
					 ni = inputs_.size();
		 i < ni;
		 ++i)
	{
		fl::InputVariable* p_input = inputs_[i];

		// check: null
		FL_DEBUG_ASSERT( p_input );

		for (std::size_t t = 0,
						 nt = p_input->numberOfTerms();
			 t < nt;
			 ++t)
		{
			fl::Term* p_term = p_input->getTerm(t);

			// check: null
			FL_DEBUG_ASSERT( p_term );

			FuzzificationNode* p_node = new FuzzificationNode(p_term, this);
			fuzzificationNodes_.push_back(p_node);

			termNodeMap[p_term] = p_node;

			// Connect every input node with its terms' node
			this->connect(varNodeMap.at(p_input), p_node);
		}
	}

	// Layer 2: complement terms layer
	// There is one node for each linguistic term of each input variable
	for (std::size_t i = 0,
					 ni = inputs_.size();
		 i < ni;
		 ++i)
	{
		fl::InputVariable* p_input = inputs_[i];

		// check: null
		FL_DEBUG_ASSERT( p_input );

		for (std::size_t t = 0,
						 nt = p_input->numberOfTerms();
			 t < nt;
			 ++t)
		{
			fl::Term* p_term = p_input->getTerm(t);

			// check: null
			FL_DEBUG_ASSERT( p_term );

			InputHedgeNode* p_node = new InputHedgeNode(fl::FactoryManager::instance()->hedge()->constructObject(Not().name()), this);
			inputHedgeNodes_.push_back(p_node);

			notFuzzificationNodeMap[p_term] = p_node;

			// Connect the term node with its negation
			this->connect(termNodeMap.at(p_term), p_node);
		}
	}

	// Layer 3: firing strength of fuzzy rules
	// There is one node for each rule
	for (std::size_t b = 0,
					 nb = ruleBlocks_.size();
		 b < nb;
		 ++b)
	{
		const fl::RuleBlock* p_ruleBlock = ruleBlocks_[b];

		// check: null
		FL_DEBUG_ASSERT( p_ruleBlock );

		if (!p_ruleBlock->isEnabled())
		{
			continue;
		}

		for (std::size_t r = 0,
						 nr = p_ruleBlock->numberOfRules();
			 r < nr;
			 ++r)
		{
			const fl::Rule* p_rule = p_ruleBlock->getRule(r);

			// check: null
			FL_DEBUG_ASSERT( p_rule );

			std::string opKeyword;
			std::vector<fl::Variable*> ruleVars;
			std::vector<fl::Term*> ruleTerms;
			std::vector<bool> ruleNots;

			detail::flattenRuleAntecedent(p_rule->getAntecedent(), ruleVars, ruleTerms, ruleNots, opKeyword);

			AntecedentNode* p_node = fl::null;
			if (opKeyword == fl::Rule::andKeyword())
			{
				p_node = new AntecedentNode(p_ruleBlock->getConjunction(), this);
			}
			else
			{
				p_node = new AntecedentNode(p_ruleBlock->getDisjunction(), this);
			}
			antecedentNodes_.push_back(p_node);

			ruleAntecedentNodeMap[p_rule] = p_node;

			// Connect every term node whose term appears in the antecedent's rule to this node
			for (std::size_t t = 0,
							 nt = ruleTerms.size();
				 t < nt;
				 ++t)
			{
				fl::Term* p_term = ruleTerms[t];

				// check: null
				FL_DEBUG_ASSERT( p_term );

				if (ruleNots[t])
				{
					this->connect(notFuzzificationNodeMap.at(p_term), p_node);
				}
				else
				{
					this->connect(termNodeMap.at(p_term), p_node);
				}
			}
		}
	}

	// Layer 4: implication of fuzzy rules
	// There is one node for each rule
	for (std::size_t b = 0,
					 nb = ruleBlocks_.size();
		 b < nb;
		 ++b)
	{
		const fl::RuleBlock* p_ruleBlock = ruleBlocks_[b];

		// check: null
		FL_DEBUG_ASSERT( p_ruleBlock );

		if (!p_ruleBlock->isEnabled())
		{
			continue;
		}

		for (std::size_t r = 0,
						 nr = p_ruleBlock->numberOfRules();
			 r < nr;
			 ++r)
		{
			const fl::Rule* p_rule = p_ruleBlock->getRule(r);

			// check: null
			FL_DEBUG_ASSERT( p_rule );

			// check: consistency (only 1 output term)
			FL_DEBUG_ASSERT( p_rule->getConsequent()->conclusions().size() == 1 );

			fl::Term* p_term = p_rule->getConsequent()->conclusions().front()->term;

			ConsequentNode* p_node = new ConsequentNode(p_term, p_ruleBlock->getActivation(), this);
			consequentNodes_.push_back(p_node);

			// Connect every input node to this node
//			for (typename std::vector<Node*>::iterator inputNodeIt = inputNodes_.begin(),
//													   inputNodeEndIt = inputNodes_.end();
//				 inputNodeIt != inputNodeEndIt;
//				 ++inputNodeIt)
//			{
//				Node* p_inpNode = *inputNodeIt;
//
//				this->connect(p_inpNode, p_node);
//			}
			// Connect the consequent of a rule with its antecedent node
			this->connect(ruleAntecedentNodeMap.at(p_rule), p_node);
		}
	}

	// Layer 5: the summation layer
	// There are two summation nodes only.
	// The first node computes the sum of the rule implications
	// (i.e., the outputs of Layer 4).
	// The second one computes the sum of the rule firing strengths (i.e., the outputs of
	// Layer 4).
	{
		AccumulationNode* p_node = fl::null;

		// Create a first summation node to compute the sum of the implication outputs
		p_node = new AccumulationNode(this);
		accumulationNodes_.push_back(p_node);
		// Connect every rule implication node to this node
		for (typename std::vector<ConsequentNode*>::iterator nodeIt = consequentNodes_.begin(),
																  nodeEndIt = consequentNodes_.end();
			 nodeIt != nodeEndIt;
			 ++nodeIt)
		{
			Node* p_consequentNode = *nodeIt;

			this->connect(p_consequentNode, p_node);
		}

		// Create a second summation node to compute the sum of all the antecedents' firing strength
		p_node = new AccumulationNode(this);
		accumulationNodes_.push_back(p_node);
		// Connect every antecedent node to this node
		for (typename std::vector<AntecedentNode*>::iterator nodeIt = antecedentNodes_.begin(),
												   nodeEndIt = antecedentNodes_.end();
			 nodeIt != nodeEndIt;
			 ++nodeIt)
		{
			Node* p_antecedentNode = *nodeIt;

			this->connect(p_antecedentNode, p_node);
		}
	}

	// Layer 6: the normalization layer
	// There is one normalization node only.
	// This node computes the ratio between the weighted sum of rules'
	// implications (i.e., the output of the first node of Layer 5) and the
	// sum of rules' firing strenghts (i.e., the output of the second node of Layer 5).
	{
		OutputNode* p_node = new OutputNode(this);
		outputNodes_.push_back(p_node);

		// Connect every summation node to this node
		for (typename std::vector<AccumulationNode*>::iterator nodeIt = accumulationNodes_.begin(),
												   nodeEndIt = accumulationNodes_.end();
			 nodeIt != nodeEndIt;
			 ++nodeIt)
		{
			Node* p_accNode = *nodeIt;

			this->connect(p_accNode, p_node);
		}
	}

//NOTE: order is valid only for Takagi-Sugeno ANFIS
//		// Set model order
//		if (dynamic_cast<fl::Constant*>(p_output_->getTerm(0)))
//		{
//			order_ = 0;
//		}
//		else if (dynamic_cast<fl::Linear*>(p_output_->getTerm(0)))
//		{
//			order_ = 1;
//		}
//[XXX]
std::cerr << "ANFIS:" << std::endl;
Engine::LayerCategory layerCat = Engine::InputLayer;
while (1)
{
    std::vector<Node*> nodes = getLayer(layerCat);
    std::cerr << "- Layer: " << layerCat << std::endl;
    std::cerr << " - #Nodes: " << nodes.size() << std::endl;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        std::cerr << " - Node #" << i << ", #inputs: " << nodes[i]->inputConnections().size() << ", #output: " << nodes[i]->outputConnections().size() << std::endl;
    }
    if (layerCat == Engine::OutputLayer)
    {
        break;
    }
    layerCat = getNextLayerCategory(layerCat);
}
//oss << std::endl << FllExporter().toString(this);
std::cerr << std::endl;
//[/XXX]

}

void Engine::connect(Node* p_from, Node* p_to)
{
	inConns_[p_to].push_back(p_from);
	outConns_[p_from].push_back(p_to);
}


}} // Namespace fl::anfis
