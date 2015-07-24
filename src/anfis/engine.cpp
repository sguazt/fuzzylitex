/**
 * \file anfis/engine.cpp
 *
 * \brief Definitions for the ANFIS engine class
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
#include <cstddef>
#include <fl/anfis/engine.h>
#include <fl/macro.h>
#include <fl/detail/math.h>
#include <fl/detail/terms.h>
#include <fl/detail/traits.h>
#include <fl/factory/FactoryManager.h>
#include <fl/factory/HedgeFactory.h>
#include <fl/fuzzylite.h>
#include <fl/hedge/Not.h>
#include <fl/norm/SNorm.h> //FIXME: needed even if not explicitly used because of fwd decl in fl::RuleBlock
#include <fl/norm/TNorm.h>
#include <fl/rule/Antecedent.h>
#include <fl/rule/Consequent.h> //FIXME: needed even if not explicitly used because of fwd decl in fl::Rule
#include <fl/rule/Expression.h>
#include <fl/rule/Rule.h>
#include <fl/rule/RuleBlock.h>
#include <fl/term/Accumulated.h> //FIXME: needed even if not explicitly used because of fwd decl in fl::OutputVariable
#include <fl/term/Term.h>
#include <fl/variable/Variable.h>
#include <fl/variable/InputVariable.h>
#include <fl/variable/OutputVariable.h>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>


namespace fl { namespace anfis {

namespace detail { namespace /*<unnamed>*/ {

void flattenRuleAntecedentRec(fl::Expression* p_expr,
                              std::vector<fl::Variable*>& vars,
                              std::vector<fl::Term*>& terms,
                              std::vector<bool>& nots,
                              std::string& opKeyword)
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

void flattenRuleAntecedent(fl::Antecedent* p_antecedent,
                           std::vector<fl::Variable*>& vars,
                           std::vector<fl::Term*>& terms,
                           std::vector<bool>& nots,
                           std::string& opKeyword)
{
    flattenRuleAntecedentRec(p_antecedent->getExpression(), vars, terms, nots, opKeyword);

    if (!terms.empty() && opKeyword.empty())
    {
        opKeyword = fl::Rule::andKeyword();
    }
}

}} // Namespace detail::<unnamed>


//////////
// Engine
//////////


Engine::Engine(const std::string& name)
: BaseType(name),
  hasBias_(false),
  isLearning_(false)
{
}

Engine::Engine(const fl::Engine& other)
: BaseType(other),
  hasBias_(false),
  isLearning_(false)
{
    this->build();
}

Engine::Engine(const Engine& other)
: BaseType(other),
  hasBias_(other.hasBias_),
  isLearning_(other.isLearning_)
{
    //// Clears the current network structure
    //this->clearAnfis();

    // Creates a new network structure
    this->build();

/*
    // Copies MF params
    // - Copies input MF params
    for (std::size_t i = 0,
                     ni = this->numberOfInputVariables();
         i < ni;
         ++i)
    {
        fl::InputVariable* p_var = this->getInputVariable(i);
        const fl::InputVariable* p_otherVar = other.getInputVariable(i);

        FL_DEBUG_ASSERT( p_var );
        FL_DEBUG_ASSERT( p_otherVar );

        for (std::size_t t = 0,
                         nt = p_var->numberOfTerms();
             t < nt;
             ++t)
        {
            fl::Term* p_term = p_var->getTerm(t);
            const fl::Term* p_otherTerm = p_otherVar->getTerm(t);

            FL_DEBUG_ASSERT( p_term );
            FL_DEBUG_ASSERT( p_otherTerm );

            const std::vector<fl::scalar> params = fl::detail::GetTermParameters(p_otherTerm);
            fl::detail::SetTermParameters(p_term, params.begin(), params.end());
        }
    }
    // - Copies output MF params
    for (std::size_t i = 0,
                     ni = this->numberOfOutputVariables();
         i < ni;
         ++i)
    {
        fl::OutputVariable* p_var = this->getOutputVariable(i);
        const fl::OutputVariable* p_otherVar = other.getOutputVariable(i);

        FL_DEBUG_ASSERT( p_var );
        FL_DEBUG_ASSERT( p_otherVar );

        for (std::size_t t = 0,
                         nt = p_var->numberOfTerms();
             t < nt;
             ++t)
        {
            fl::Term* p_term = p_var->getTerm(t);
            const fl::Term* p_otherTerm = p_otherVar->getTerm(t);

            FL_DEBUG_ASSERT( p_term );
            FL_DEBUG_ASSERT( p_otherTerm );

            const std::vector<fl::scalar> params = fl::detail::GetTermParameters(p_otherTerm);
            fl::detail::SetTermParameters(p_term, params.begin(), params.end());
        }
    }
*/
}

Engine::~Engine()
{
    this->clear();
}

Engine& Engine::operator=(const Engine& rhs)
{
    if (this != &rhs)
    {
        this->clear();

        BaseType::operator=(rhs);

//      Engine tmp(rhs);
//
//      std::swap(inputNodes_, tmp.inputNodes_);
//      std::swap(fuzzificationNodes_, tmp.fuzzificationNodes_);
//      std::swap(inputHedgeNodes_, tmp.inputHedgeNodes_);
//      std::swap(antecedentNodes_, tmp.antecedentNodes_);
//      std::swap(consequentNodes_, tmp.consequentNodes_);
//      std::swap(accumulationNodes_, tmp.accumulationNodes_);
//      std::swap(outputNodes_, tmp.outputNodes_);
//      std::swap(inConns_, tmp.inConns_);
//      std::swap(outConns_, tmp.outConns_);
//      std::swap(hasBias_, tmp.hasBias_);
//      //std::swap(bias_, tmp.bias_);
//      std::swap(isLearning_, tmp.isLearning_);

        this->build();

        this->updateAnfisReferences();
    }

    return *this;
}

Engine* Engine::clone() const
{
    return new Engine(*this);
}

//void Engine::swap(Engine& other)
//{
//  std::swap(inputNodes_, other.inputNodes_);
//  std::swap(fuzzificationNodes_, other.fuzzificationNodes_);
//  std::swap(inputHedgeNodes_, other.inputHedgeNodes_);
//  std::swap(antecedentNodes_, other.antecedentNodes_);
//  std::swap(consequentNodes_, other.consequentNodes_);
//  std::swap(accumulationNodes_, other.accumulationNodes_);
//  std::swap(outputNodes_, other.outputNodes_);
//  std::swap(inConns_, other.inConns_);
//  std::swap(outConns_, other.outConns_);
//  this->updateAnfisReferences();
//  other.updateAnfisReferences();
//}

//TODO
//std::string Engine::toString() const
//{
//  return FllExporter().toString(this);
//}

//TODO
//void Engine::configure(fl::TNorm* conjunction, fl::SNorm* disjunction,
//                     fl::TNorm* activation, fl::SNorm* accumulation,
//                     fl::Defuzzifier* defuzzifier)
//{
//  throw std::runtime_error("Engine::configure to be implemented");
//} 

//TODO
//void Engine::configure(const std::string& conjunctionT,
//                     const std::string& disjunctionS,
//                     const std::string& activationT,
//                     const std::string& accumulationS,
//                     const std::string& defuzzifier,
//                     int resolution)
//{
//  throw std::runtime_error("Engine::configure to be implemented");
//}

bool Engine::isReady(std::string* p_status) const
{
    bool ready = BaseType::isReady(p_status);

    if (ready)
    {
        std::ostringstream oss;
        if (inputNodes_.empty()
            || fuzzificationNodes_.empty()
            || inputHedgeNodes_.empty()
            || antecedentNodes_.empty()
            || consequentNodes_.empty()
            || accumulationNodes_.empty()
            || outputNodes_.empty()
            || inConns_.empty()
            || outConns_.empty())
        {
            oss << "- Engine <" << this->getName() << "> has an incomplete ANFIS model" << std::endl;
            ready = false;
        }
        else if (inputNodes_.size() != this->numberOfInputVariables())
        {
            oss << "- Engine <" << this->getName() << "> has a bad number of input nodes in the ANFIS model" << std::endl;
            ready = false;
        }
        else if (outputNodes_.size() != this->numberOfOutputVariables())
        {
            oss << "- Engine <" << this->getName() << "> has a bad number of output nodes in the ANFIS model" << std::endl;
            ready = false;
        }
    }

    return ready;
}

void Engine::process()
{
	//// Necessary to be fully compatible with classic Fuzzy engine (indeed, fl::anfis::Engine is a fl::Engine)
	//// Specifilly, the call to BaseType::process activates rule blocks and defuzzify output variables
	//BaseType::process();

	for (std::size_t i = 0,
					 ni = this->outputVariables().size();
		 i < ni;
		 ++i)
	{
		this->getOutputVariable(i)->fuzzyOutput()->clear();
	}

	for (std::size_t i = 0,
					 ni = this->ruleBlocks().size();
		 i < ni;
		 ++i)
	{
		fl::RuleBlock* ruleBlock = this->getRuleBlock(i);
		if (ruleBlock->isEnabled())
		{
			ruleBlock->activate();
		}
	}

    this->eval();
}

void Engine::restart()
{
    // Invalidate input and output variables
    BaseType::restart();
    // Invalidate ANFIS nodes
    this->eval();
}

void Engine::setHasBias(bool value)
{
    hasBias_ = value;
}

bool Engine::hasBias() const
{
    return hasBias_;
}

void Engine::setBias(const std::vector<fl::scalar>& value)
{
    this->setBias(value.begin(), value.end());
}

std::vector<fl::scalar> Engine::getBias() const
{
    const std::size_t n = outputNodes_.size();

    std::vector<fl::scalar> res(n, 0);

    for (std::size_t i = 0; i < n; ++i)
    {
        res[i] = outputNodes_[i]->getBias();
    }

    return res;
}

void Engine::setIsLearning(bool value)
{
    isLearning_ = value;
}

bool Engine::isLearning() const
{
    return isLearning_;
}

std::vector<fl::scalar> Engine::getInputValues() const
{
    const std::size_t n = inputNodes_.size();

    std::vector<fl::scalar> inputs(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        inputs[i] = inputNodes_[i]->getInputVariable()->getValue();
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
    this->clearAnfis();

    //TODO: it would be great if fl::Engine provide a 'clear' method as well...

//  for (std::size_t i = 0,
//                   n = ruleBlocks_.size();
//       i < n;
//       ++i)
//  {
//      delete ruleBlocks_[i];
//  }
//  ruleBlocks_.clear();
    while (this->numberOfRuleBlocks() > 0)
    {
        fl::RuleBlock* p_block = this->removeRuleBlock(0);
        if (p_block)
        {
            delete p_block;
        }
    }

//  for (std::size_t i = 0,
//                   n = outputs_.size();
//       i < n;
//       ++i)
//  {
//      delete outputs_[i];
//  }
//  outputs_.clear();
    while (this->numberOfOutputVariables() > 0)
    {
        fl::OutputVariable* p_var = this->removeOutputVariable(0);
        if (p_var)
        {
            delete p_var;
        }
    }

//  for (std::size_t i = 0,
//                   n = inputs_.size();
//       i < n;
//       ++i)
//  {
//      delete inputs_[i];
//  }
//  inputs_.clear();
    while (this->numberOfInputVariables() > 0)
    {
        fl::InputVariable* p_var = this->removeInputVariable(0);
        if (p_var)
        {
            delete p_var;
        }
    }
}

void Engine::clearAnfis()
{
    //hasBias_ = false;
    //bias_.clear();
    //isLearning_ = false;

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

void Engine::updateAnfisReferences()
{
    //BaseType::updateReferences();

    for (std::size_t i = 0,
                     n = inputNodes_.size();
         i < n;
         ++i)
    {
        inputNodes_[i]->setEngine(this);
    }
    for (std::size_t i = 0,
                     n = fuzzificationNodes_.size();
         i < n;
         ++i)
    {
        fuzzificationNodes_[i]->setEngine(this);
    }
    for (std::size_t i = 0,
                     n = inputHedgeNodes_.size();
         i < n;
         ++i)
    {
        inputHedgeNodes_[i]->setEngine(this);
    }
    for (std::size_t i = 0,
                     n = antecedentNodes_.size();
         i < n;
         ++i)
    {
        antecedentNodes_[i]->setEngine(this);
    }
    for (std::size_t i = 0,
                     n = consequentNodes_.size();
         i < n;
         ++i)
    {
        consequentNodes_[i]->setEngine(this);
    }
    for (std::size_t i = 0,
                     n = accumulationNodes_.size();
         i < n;
         ++i)
    {
        accumulationNodes_[i]->setEngine(this);
    }
    for (std::size_t i = 0,
                     n = outputNodes_.size();
         i < n;
         ++i)
    {
        outputNodes_[i]->setEngine(this);
    }
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
    // Check inference type
    if (this->type() != fl::Engine::TakagiSugeno
        && this->type() != fl::Engine::Tsukamoto)
    {
        FL_THROW2(std::logic_error, "ANFIS must represent either a Takagi-Sugeno or a Tsukamoto fuzzy inference system");
    }

    // Check input vars
    if (this->numberOfInputVariables() == 0)
    {
        FL_THROW2(std::logic_error, "Not enough input variables to build an ANFIS model");
    }

    // Check output vars
    if (this->numberOfOutputVariables() == 0)
    {
        FL_THROW2(std::logic_error, "Not enough output variables to build an ANFIS model");
    }

    // Check rules
    {
        std::size_t numRules = 0;
        for (std::size_t r = 0,
                         nr = this->numberOfRuleBlocks();
             r < nr && numRules == 0;
             ++r)
        {
            const fl::RuleBlock* rb = this->getRuleBlock(r);

            if (rb->isEnabled())
            {
                numRules += rb->numberOfRules();
            }
        }

        if (numRules == 0)
        {
            FL_THROW2(std::logic_error, "Not enough enabled rules to build an ANFIS model");
        }
    }
}

void Engine::build()
{
    // Clears the current network structure
    this->clearAnfis();

    // Check consistency
    this->check();

    // Build...

    std::map<const fl::Variable*,Node*> varNodeMap;
    std::map<const fl::Term*,Node*> termNodeMap;
    std::map<const fl::Term*,Node*> notFuzzificationNodeMap;
    std::map<const fl::Rule*,Node*> ruleAntecedentNodeMap;
    std::map<const fl::Variable*,std::vector<Node*> > varConsequentNodesMap;

    // Layer 0 (the input layer): input linguistic variables
    // There is one node for each input variable
    for (std::size_t i = 0,
                     n = this->numberOfInputVariables();
         i < n;
         ++i)
    {
        fl::InputVariable* p_input = this->getInputVariable(i);

        // check: null
        FL_DEBUG_ASSERT( p_input );

        InputNode* p_node = new InputNode(p_input, this);
        inputNodes_.push_back(p_node);

        varNodeMap[p_input] = p_node;
    }

    // Layer 1 (fuzzification layer): linguistic terms layer
    // There is one node for each linguistic term of each input variable
    for (std::size_t i = 0,
                     ni = this->numberOfInputVariables();
         i < ni;
         ++i)
    {
        fl::InputVariable* p_input = this->getInputVariable(i);

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

    // Layer 2 (input hedge layer): complement terms layer
    // There is one node for each linguistic term of each input variable
    for (std::size_t i = 0,
                     ni = this->numberOfInputVariables();
         i < ni;
         ++i)
    {
        fl::InputVariable* p_input = this->getInputVariable(i);

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

    // Layer 3 (antecedent layer): firing strength of fuzzy rules
    // There is one node for each rule
    for (std::size_t b = 0,
                     nb = this->numberOfRuleBlocks();
         b < nb;
         ++b)
    {
        const fl::RuleBlock* p_ruleBlock = this->getRuleBlock(b);

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

    // Layer 4 (consequent layer): implication of fuzzy rules
    // There is one node for each rule and each output term in rule consequent
    for (std::size_t b = 0,
                     nb = this->numberOfRuleBlocks();
         b < nb;
         ++b)
    {
        const fl::RuleBlock* p_ruleBlock = this->getRuleBlock(b);

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

            //// check: consistency (only 1 output term)
            //FL_DEBUG_ASSERT( p_rule->getConsequent()->conclusions().size() == 1 );

            const std::vector<fl::Proposition*> conclusions =   p_rule->getConsequent()->conclusions();
            for (std::size_t c = 0,
                             nc = conclusions.size();
                 c < nc;
                 ++c)
            {
                fl::Proposition* p_prop = conclusions.at(c);

                // check: null
                FL_DEBUG_ASSERT( p_prop );

                fl::Term* p_term = p_prop->term;

                ConsequentNode* p_node = new ConsequentNode(p_term, /*p_ruleBlock->getActivation(),*/ this);
                consequentNodes_.push_back(p_node);

                varConsequentNodesMap[p_prop->variable].push_back(p_node);

                // Connect every input node to this node
//              for (typename std::vector<Node*>::iterator inputNodeIt = inputNodes_.begin(),
//                                                         inputNodeEndIt = inputNodes_.end();
//                   inputNodeIt != inputNodeEndIt;
//                   ++inputNodeIt)
//              {
//                  Node* p_inpNode = *inputNodeIt;
//
//                  this->connect(p_inpNode, p_node);
//              }
                // Connect the consequent of a rule with its antecedent node
                this->connect(ruleAntecedentNodeMap.at(p_rule), p_node);
            }
        }
    }

    // Layer 5 (accumulation layer): the summation layer
    // There are two summation nodes only.
    // The first nodes compute the sum of the rule implications for each output variable
    // (i.e., the outputs of Layer 4).
    // The last node computes the sum of the rule firing strengths (i.e., the outputs of
    // Layer 4).
    {
        // Create a first set of summation nodes to compute the sum of the implication outputs
        for (std::size_t i = 0,
                         ni = this->numberOfOutputVariables();
             i < ni;
             ++i)
        {
            const fl::OutputVariable* p_var = this->getOutputVariable(i);

            AccumulationNode* p_node = new AccumulationNode(this);
            accumulationNodes_.push_back(p_node);
            // Connect every rule implication node for this output variable to this node
//          for (typename std::vector<ConsequentNode*>::iterator nodeIt = consequentNodes_.begin(),
//                                                               nodeEndIt = consequentNodes_.end();
//               nodeIt != nodeEndIt;
//               ++nodeIt)
//          {
//              Node* p_consequentNode = *nodeIt;
//
//              this->connect(p_consequentNode, p_node);
//          }
            if (varConsequentNodesMap.count(p_var) > 0)
            {
                for (std::size_t j = 0,
                                 nj = varConsequentNodesMap.at(p_var).size();
                     j < nj;
                     ++j)
                {
                    Node* p_consequentNode = varConsequentNodesMap.at(p_var).at(j);

                    this->connect(p_consequentNode, p_node);
                }
            }
        }

        // Create a second summation node to compute the sum of all the antecedents' firing strength
        AccumulationNode* p_node = new AccumulationNode(this);
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

    // Layer 6 (output layer): the normalization layer
    // There is one normalization node only.
    // This node computes the ratio between the weighted sum of rules'
    // implications (i.e., the output of the first node of Layer 5) and the
    // sum of rules' firing strenghts (i.e., the output of the second node of Layer 5).
    for (std::size_t i = 0,
                     n = this->numberOfOutputVariables();
         i < n;
         ++i)
    {
        fl::OutputVariable* p_output = this->getOutputVariable(i);

        // check: null
        FL_DEBUG_ASSERT( p_output );

        OutputNode* p_node = new OutputNode(p_output, this);
        outputNodes_.push_back(p_node);

//      // Connect every summation node to this node
//      for (typename std::vector<AccumulationNode*>::iterator nodeIt = accumulationNodes_.begin(),
//                                                             nodeEndIt = accumulationNodes_.end();
//           nodeIt != nodeEndIt;
//           ++nodeIt)
//      {
//          Node* p_accNode = *nodeIt;
//
//          this->connect(p_accNode, p_node);
//      }

        // check: the accumulation layer must have a specific node (other than the sum of the all firing strengths) for this output variable
        FL_DEBUG_ASSERT( i < (accumulationNodes_.size()-1) );

        // Connect the corresponding summation node (i.e., the sum of the firing strengths of its output terms) to this node
        this->connect(accumulationNodes_[i], p_node);
        // Connect the last summation node (i.e., the sum of all firing strengths) to this node
        this->connect(accumulationNodes_.back(), p_node);
    }

//NOTE: order is valid only for Takagi-Sugeno ANFIS
//      // Set model order
//      if (dynamic_cast<fl::Constant*>(p_output_->getTerm(0)))
//      {
//          order_ = 0;
//      }
//      else if (dynamic_cast<fl::Linear*>(p_output_->getTerm(0)))
//      {
//          order_ = 1;
//      }
//[XXX]
//std::cerr << "ANFIS:" << std::endl;
//Engine::LayerCategory layerCat = Engine::InputLayer;
//while (1)
//{
//    std::vector<Node*> nodes = getLayer(layerCat);
//    std::cerr << "- Layer: " << layerCat << std::endl;
//    std::cerr << " - #Nodes: " << nodes.size() << std::endl;
//    for (std::size_t i = 0; i < nodes.size(); ++i)
//    {
//        std::cerr << " - Node #" << i << ", #inputs: " << nodes[i]->inputConnections().size() << ", #output: " << nodes[i]->outputConnections().size() << std::endl;
//    }
//    if (layerCat == Engine::OutputLayer)
//    {
//        break;
//    }
//    layerCat = getNextLayerCategory(layerCat);
//}
////oss << std::endl << FllExporter().toString(this);
//std::cerr << std::endl;
//[/XXX]

}

void Engine::connect(Node* p_from, Node* p_to)
{
    inConns_[p_to].push_back(p_from);
    outConns_[p_from].push_back(p_to);
}


}} // Namespace fl::anfis
