#include <cstddef>
#include <fl/anfis/engine.h>
#include <fl/anfis/nodes.h>
#include <fl/commons.h>
#include <fl/detail/math.h>
#include <fl/detail/terms.h>
#include <fl/detail/traits.h>
#include <fl/hedge/Hedge.h>
#include <fl/norm/Norm.h>
#include <fl/norm/s/AlgebraicSum.h>
#include <fl/norm/s/Maximum.h>
#include <fl/norm/TNorm.h>
#include <fl/norm/t/AlgebraicProduct.h>
#include <fl/norm/t/Minimum.h>
#include <fl/term/Term.h>
#include <fl/variable/InputVariable.h>
#include <fl/variable/OutputVariable.h>
#include <stdexcept>
#include <vector>


namespace fl { namespace anfis {

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

OutputNode::OutputNode(fl::OutputVariable* p_var, Engine* p_engine)
: Node(p_engine),
  p_var_(p_var),
  bias_(0)
{
}

void OutputNode::setBias(fl::scalar value)
{
	bias_ = value;
}

fl::scalar OutputNode::getBias() const
{
	return bias_;
}

fl::scalar OutputNode::doEval()
{
	std::vector<fl::scalar> inputs = this->inputs();

	if (inputs.size() != 2)
	{
		FL_THROW2(std::logic_error, "Output node must have exactly two inputs");
	}

	fl::scalar res = 0;

	if (fl::detail::FloatTraits<fl::scalar>::ApproximatelyZero(inputs[1]))
	{
		// Handle zero firing strength...

		if (this->getEngine()->isLearning())
		{
			// The training algorithm will take care of this...
			res = fl::nan;
		}
		else if (this->getEngine()->hasBias())
		{
			// Return the stored bias
			res = bias_;
		}
		else
		{
			// Return the average value among all possible values
			res = (p_var_->getMinimum()+p_var_->getMaximum())/2.0;
		}
	}
	else
	{
		res = inputs[0]/inputs[1];
	}

	p_var_->setOutputValue(res);

	return res;
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

}} // Namespace fl::anfis
