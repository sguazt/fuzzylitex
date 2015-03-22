#include <fl/anfis/model.h>
#include <fl/commons.h>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/Headers.h>


namespace fl { namespace anfis {

namespace detail { namespace /*<unnamed>*/ {

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

}} // Namespace detail::<unnamed>


////////////////////////////////////////////////////////////////////////////////
// Nodes
////////////////////////////////////////////////////////////////////////////////

/////////////
// Node
/////////////


Node::Node(Model* p_model)
: p_model_(p_model)
{
}

Node::~Node()
{
	// empty
}

void Node::setModel(Model* p_model)
{
	p_model_ = p_model;
}

Model* Node::getModel() const
{
	return p_model_;
}

std::vector<Node*> Node::inputConnections() const
{
	return p_model_->inputConnections(this);
}

std::vector<Node*> Node::outputConnections() const
{
	return p_model_->outputConnections(this);
}

std::vector<fl::scalar> Node::inputs() const
{
	std::vector<fl::scalar> inps;

	std::vector<Node*> inConns = p_model_->inputConnections(this);
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

fl::scalar Node::getValue() const
{
	return val_;
}

void Node::setValue(fl::scalar v)
{
	val_ = v;
}

InputNode::InputNode(fl::InputVariable* p_var, Model* p_model)
: Node(p_model),
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

/////////////
// Term Node
/////////////

TermNode::TermNode(fl::Term* p_term, Model* p_model)
: Node(p_model),
  p_term_(p_term)
{
}

fl::Term* TermNode::getTerm() const
{
	return p_term_;
}

fl::scalar TermNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Term node must have exactly one input");
	}

	return p_term_->membership(inputs.front());
}


HedgeNode::HedgeNode(fl::Hedge* p_hedge, Model* p_model)
: Node(p_model),
  p_hedge_(p_hedge)
{
}

fl::Hedge* HedgeNode::getHedge() const
{
	return p_hedge_;
}

fl::scalar HedgeNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Hedge node must have exactly one input");
	}

	return p_hedge_->hedge(inputs.front());
}

RuleFiringStrengthNode::RuleFiringStrengthNode(fl::Norm* p_norm, Model* p_model)
: Node(p_model),
  p_norm_(p_norm)
{
}

fl::Norm* RuleFiringStrengthNode::getNorm() const
{
	return p_norm_;
}

fl::scalar RuleFiringStrengthNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

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

/*
private: std::vector<fl::scalar> RuleFiringStrengthNode::doEvalDerivativeWrtInput(ForwardIterator inputFirst, ForwardIterator inputLast,
														  ForwardIterator weightFirst, ForwardIterator weightLast) const
{
	if (dynamic_cast<fl::Minimum*>(p_norm_)
		|| dynamic_cast<fl::Maximum*>(p_norm_))
	{
		// The derivative is 0 for all inputs, excepts for the one
		// corresponding to the min (or max) for which the derivative is the
		// associated weight

		const fl::scalar fx = this->eval(inputFirst, inputLast, weightFirst, weightLast);

		std::vector<fl::scalar> res;
		while (inputFirst != inputLast && weightFirst != weightLast)
		{
			const fl::scalar in = *inputFirst;
			const fl::scalar w = *weightFirst;
			const fl::scalar val = w*in;

			if (fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(fx, val))
			{
				res.push_back(w);
			}
			else
			{
				res.push_back(0);
			}

			++inputFirst;
			++weightFirst;
		}

		return res;
	}
	else if (dynamic_cast<fl::AlgebraicProduct*>(p_norm_))
	{
		// The derivative for a certain input i is given by the product of
		// all the weights and inputs but i

		std::vector<fl::scalar> res;

		ForwardIterator inputIt = inputFirst;
		ForwardIterator weightIt = weightFirst;
		while (inputFirst != inputLast && weightFirst != weightLast)
		{
			fl::scalar prod = 1;
			while (inputIt != inputLast && weightIt != weightLast)
			{
				const fl::scalar w = *weightIt;

				if (inputIt != inputFirst)
				{
					const fl::scalar in = *inputIt;

					prod *= in
				}
				prod *= w;

				++inputIt;
				++weightIt;
			}

			res.push_back(prod);

			++inputFirst;
			++weightFirst;
		}

		return res;
	}
	else if (dynamic_cast<fl::AlgebraicSum*>(p_norm_))
	{
		// The derivative of each input is given by:
		//  \frac{\partial [x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3 + x_4 - (x_1 + x_2 - (x_1*x_2) + x_3 - (x_1 + x_2 - (x_1*x_2))*x_3)*x_4 + ...]}{\partial x_1}
		//  = 1 - x_2 - x_3 - x_2*x_3 - x_4 - x_2*x_4 - x_3*x_4 - x_2*x_3*x_4 + ...
		//  = (1-x2)*(1-x3)*(1-x4)*...
		// In order to keep into account weights we have:
		//  \frac{\partial [w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2) + w_3*x_3 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2))*w_3*x_3 + w_4*x_4 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2) + w_3*x_3 - (w_1*x_1 + w_2*x_2 - (w_1*x_1*w_2*x_2))*w_3*x_3)*w_4*x_4 + ...]}{\partial x_1}
		//  = w_1*(1-w_2*x_2)*(1-w_3*x_3)*(1-w_4*x_4)*...

		std::vector<fl::scalar> res;

		ForwardIterator inputIt = inputFirst;
		ForwardIterator weightIt = weightFirst;
		while (inputFirst != inputLast && weightFirst != weightLast)
		{
			fl::scalar prod = 1;
			while (inputIt != inputLast && weightIt != weightLast)
			{
				const fl::scalar w = *weightIt;

				if (inputIt != inputFirst)
				{
					const fl::scalar in = *inputIt;

					prod *= (1-in);
				}
				prod *= w;

				++inputIt;
				++weightIt;
			}

			res.push_back(prod);

			++inputFirst;
			++weightFirst;
		}

		return res;
	}
	else
	{
		FL_THROW2(std::runtime_error, "Norm operator '" + p_norm_->className() + "' not yet implemented");
	}
}
*/

RuleImplicationNode::RuleImplicationNode(fl::Term* p_term, fl::TNorm* p_tnorm, Model* p_model)
: Node(p_model),
  p_term_(p_term),
  p_tnorm_(p_tnorm)
{
}

fl::Term* RuleImplicationNode::getTerm() const
{
	return p_term_;
}

fl::TNorm* RuleImplicationNode::getTNorm() const
{
	return p_tnorm_;
}

fl::scalar RuleImplicationNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 1)
	{
		FL_THROW2(std::logic_error, "Rule implication node must have exactly one input");
	}

	return p_tnorm_->compute(inputs.front(), p_term_->membership(1.0));
}

SumNode::SumNode(Model* p_model)
: Node(p_model)
{
}

fl::scalar SumNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

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

NormalizationNode::NormalizationNode(Model* p_model)
: Node(p_model)
{
}

fl::scalar NormalizationNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 2)
	{
		FL_THROW2(std::logic_error, "Normalization node must have exactly two inputs");
	}

	return inputs[0]/inputs[1];
}


/////////
// Model
/////////

Model::Model()
{
}

Model::~Model()
{
	this->clear();
}

std::vector<fl::scalar> Model::evalInputLayer()
{
	return this->evalLayer(inputNodes_.begin(), inputNodes_.end());
}

std::vector<fl::scalar> Model::evalFuzzificationLayer()
{
	return this->evalLayer(inputTermNodes_.begin(), inputTermNodes_.end());
}

std::vector<fl::scalar> Model::evalHedgeLayer()
{
	return this->evalLayer(inputHedgeNodes_.begin(), inputHedgeNodes_.end());
}

std::vector<fl::scalar> Model::evalAntecedentLayer()
{
	return this->evalLayer(antecedentNodes_.begin(), antecedentNodes_.end());
}

std::vector<fl::scalar> Model::evalConsequentLayer()
{
	return this->evalLayer(consequentNodes_.begin(), consequentNodes_.end());
}

std::vector<fl::scalar> Model::evalAccumulationLayer()
{
	return this->evalLayer(sumNodes_.begin(), sumNodes_.end());
}

std::vector<fl::scalar> Model::evalNormalizationLayer()
{
	return this->evalLayer(inferenceNodes_.begin(), inferenceNodes_.end());
}

std::vector<fl::scalar> Model::eval()
{
	// Eval input layer
	this->evalInputLayer();
	// Eval fuzzification layer
	this->evalFuzzificationLayer();
	// Eval hedge layer
	this->evalHedgeLayer();
	// Eval rule antecedent layer
	this->evalAntecedentLayer();
	// Eval rule consequent layer
	this->evalConsequentLayer();
	// Eval rule accumulation layer
	this->evalAccumulationLayer();
	// Eval rule strength normalization layer
	return this->evalNormalizationLayer();
}

void Model::clear()
{
	inConns_.clear();
	outConns_.clear();

	for (std::size_t i = 0,
					 n = inputNodes_.size();
		 i < n;
		 ++i)
	{
		delete inputNodes_[i];
	}
	inputNodes_.clear();

	for (std::size_t i = 0,
					 n = inputTermNodes_.size();
		 i < n;
		 ++i)
	{
		delete inputTermNodes_[i];
	}
	inputTermNodes_.clear();

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
					 n = sumNodes_.size();
		 i < n;
		 ++i)
	{
		delete sumNodes_[i];
	}
	sumNodes_.clear();

	for (std::size_t i = 0,
					 n = inferenceNodes_.size();
		 i < n;
		 ++i)
	{
		delete inferenceNodes_[i];
	}
	inferenceNodes_.clear();
}

std::vector<Node*> Model::inputConnections(const Node* p_node) const
{
	return inConns_.at(p_node);
}

std::vector<Node*> Model::outputConnections(const Node* p_node) const
{
	return outConns_.at(p_node);
}

void Model::check()
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

void Model::build()
{
	this->check();

	std::map<const fl::Variable*,Node*> varNodeMap;
	std::map<const fl::Term*,Node*> termNodeMap;
	std::map<const fl::Term*,Node*> notTermNodeMap;
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

			TermNode* p_node = new TermNode(p_term, this);
			inputTermNodes_.push_back(p_node);

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

			HedgeNode* p_node = new HedgeNode(fl::FactoryManager::instance()->hedge()->constructObject(Not().name()), this);
			inputHedgeNodes_.push_back(p_node);

			notTermNodeMap[p_term] = p_node;

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

			RuleFiringStrengthNode* p_node = fl::null;
			if (opKeyword == fl::Rule::andKeyword())
			{
				p_node = new RuleFiringStrengthNode(p_ruleBlock->getConjunction(), this);
			}
			else
			{
				p_node = new RuleFiringStrengthNode(p_ruleBlock->getDisjunction(), this);
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
					this->connect(notTermNodeMap.at(p_term), p_node);
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

			RuleImplicationNode* p_node = new RuleImplicationNode(p_term, p_ruleBlock->getActivation(), this);
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
		SumNode* p_node = fl::null;

		// Create a first summation node to compute the sum of the implication outputs
		p_node = new SumNode(this);
		sumNodes_.push_back(p_node);
		// Connect every rule implication node to this node
		for (typename std::vector<RuleImplicationNode*>::iterator nodeIt = consequentNodes_.begin(),
																  nodeEndIt = consequentNodes_.end();
			 nodeIt != nodeEndIt;
			 ++nodeIt)
		{
			Node* p_consequentNode = *nodeIt;

			this->connect(p_consequentNode, p_node);
		}

		// Create a second summation node to compute the sum of all the antecedents' firing strength
		p_node = new SumNode(this);
		sumNodes_.push_back(p_node);
		// Connect every antecedent node to this node
		for (typename std::vector<RuleFiringStrengthNode*>::iterator nodeIt = antecedentNodes_.begin(),
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
		NormalizationNode* p_node = new NormalizationNode(this);
		inferenceNodes_.push_back(p_node);

		// Connect every summation node to this node
		for (typename std::vector<SumNode*>::iterator nodeIt = sumNodes_.begin(),
												   nodeEndIt = sumNodes_.end();
			 nodeIt != nodeEndIt;
			 ++nodeIt)
		{
			Node* p_sumNode = *nodeIt;

			this->connect(p_sumNode, p_node);
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
}

void Model::connect(Node* p_from, Node* p_to)
{
	inConns_[p_to].push_back(p_from);
	outConns_[p_from].push_back(p_to);
}


}} // Namespace fl::anfis
