/**
 * \file fl/anfis/engine.tpp
 *
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly.
 */

#ifndef FL_ANFIS_ENGINE_TPP
#define FL_ANFIS_ENGINE_TPP


#include <fl/commons.h>
#include <fl/rule/RuleBlock.h>
#include <fl/variable/InputVariable.h>
#include <fl/variable/OutputVariable.h>
#include <stdexcept>
#include <string>
#include <vector>


namespace fl { namespace anfis {

template <typename InputIterT,
		  typename OutputIterT,
		  typename RuleBlockIterT>
Engine::Engine(InputIterT inputFirst, InputIterT inputLast,
			   OutputIterT outputFirst, OutputIterT outputLast,
			   RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast,
			   const std::string& name)
{
	this->setInputVariables(inputFirst, inputLast);
	this->setOutputVariables(outputFirst, outputLast);
	this->setRuleBlocks(ruleBlockFirst, ruleBlockLast);
	this->setName(name);
	this->setHasBias(false);

	this->build();
}

template <typename IterT>
void Engine::setInputVariables(IterT first, IterT last)
{
	//inputs_.clear();
	//inputs_.assign(first, last);
	BaseType::setInputVariables(std::vector<fl::InputVariable*>());
	while (first != last)
	{
		this->addInputVariable(*first);
		++first;
	}
}

template <typename IterT>
void Engine::setOutputVariables(IterT first, IterT last)
{
	//outputs_.clear();
	//outputs_.assign(first, last);
	BaseType::setOutputVariables(std::vector<fl::OutputVariable*>());
	while (first != last)
	{
		this->addOutputVariable(*first);
		++first;
	}
}

template <typename IterT>
void Engine::setRuleBlocks(IterT first, IterT last)
{
	//ruleBlocks_.clear();
	//ruleBlocks_.assign(first, last);
	BaseType::setRuleBlocks(std::vector<fl::RuleBlock*>());
	while (first != last)
	{
		this->addRuleBlock(*first);
		++first;
	}
}

template <typename InputIterT,
		  typename OutputIterT,
		  typename RuleBlockIterT>
void Engine::build(InputIterT inputFirst, InputIterT inputLast,
				   OutputIterT outputFirst, OutputIterT outputLast,
				   RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast)
{
	this->clear();

	//inputs_.assign(inputFirst, inputLast);
	this->setInputVariables(inputFirst, inputLast);
	//outputs_.assign(outputFirst, outputLast);
	this->setOutputVariables(outputFirst, outputLast);
	//ruleBlocks_.assign(ruleBlockFirst, ruleBlockLast);
	this->setRuleBlocks(ruleBlockFirst, ruleBlockLast);

	this->build();
}

template <typename IterT>
void Engine::setBias(IterT first, IterT last)
{
	const std::size_t ni = outputNodes_.size();

	std::size_t i = 0;
	while (first != last && i < ni)
	{
		outputNodes_[i]->setBias(*first);

		++first;
		++i;
	}

	//FIXME: decide if the remaining biases should be set to zero or left untouched
	//for (; i < ni; ++i)
	//{
	//	outputNodes_[i]->setBias(0);
	//}
}

template <typename IterT>
void Engine::setInputValues(IterT first, IterT last)
{
	std::vector<InputNode*>::iterator nodeIt = inputNodes_.begin();
	std::vector<InputNode*>::iterator nodeEndIt = inputNodes_.end();

	while (first != last && nodeIt != nodeEndIt)
	{
		(*nodeIt)->getInputVariable()->setValue(*first);

		++first;
		++nodeIt;
	}

	if (first != last || nodeIt != nodeEndIt)
	{
		FL_THROW2(std::invalid_argument, "Wrong number of inputs");
	}
}

template <typename IterT>
std::vector<fl::scalar> Engine::eval(IterT first, IterT last)
{
	this->setInputValues(first, last);
	return this->eval();
}

template <typename IterT>
std::vector<fl::scalar> Engine::evalTo(IterT first, IterT last, LayerCategory layer)
{
	this->setInputValues(first, last);
	return this->evalTo(layer);
}

template <typename IterT>
std::vector<fl::scalar> Engine::evalLayer(IterT first, IterT last)
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

}} // Namespace fl::anfis

#endif // FL_ANFIS_ENGINE_TPP
