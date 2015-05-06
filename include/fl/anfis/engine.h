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
#include <fl/macro.h>
#include <fl/rule/RuleBlock.h>
#include <fl/variable/InputVariable.h>
#include <fl/variable/OutputVariable.h>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>


namespace fl { namespace anfis {

/**
 * The Adaptive Neuro-Fuzzy Inference System (ANFIS) engine
 *
 * This class implements the <em>Adaptive Neuro-Fuzzy Inference System</em>
 * (ANFIS) model proposed by Jang et al in [Jang1993,Jang1997].
 *
 * Currently, the ANFIS model is only implemented for the Takagi-Sugeno-Kang
 * and Tsukamoto fuzzy inference system.
 *
 * Essentially, an ANFIS model is an adaptive network (generalized neural
 * network) where there are two types of nodes: fixed nodes (such that each node
 * function is fixed) and adaptive nodes (such that each node function depends
 * by a set of parameters).
 * The nodes of the network is organized in layers:
 * -# <em>Input layer</em>: this is the input layer of the network.
 *  There is a fixed node for each input variable.
 * -# <em>Fuzzification layer</em>: this is the second layer of the
 *  network used to compute membership degree of inputs.
 *  There is an adaptive node for each input term.
 * -# <em>Input hedge layer</em>: this is the third layer of the network
 *  used to apply hedges to inputs.
 *  There is a fixed node for each input hedge.
 *  Currently, only the <em>not</em> hedge is supported.
 * -# <em>Antecedent layer</em>: this is the fourth layer of the network
 *  used to compute the rule firing strength.
 *  There is a fixed node for each rule.
 * -# <em>Consequent layer</em>: this is the fifth layer of the network
 *  used to perform fuzzy implication.
 *  There is an adaptive node for each rule.
 *  Currently, only fuzzy terms compatible with the Takagi-Sugeno-Kang and
 *  Tsukamoto fuzzy inference systems are supported.
 * -# <em>Accumulation layer</em>: this is the sixth layer of the network
 *  used to aggregate the output of the rules.
 *  There is a fixed node for each output variable.
 * -# <em>Output layer</em>: this is the last layer of the network used to
 *  compute the final (defuzzified) output value.
 *  There is a fixed node for each output variable.
 * .
 *
 * References:
 * -# [Jang1993] J.-S.R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference Systems," IEEE Transactions on Systems, Man, and Cybernetics, 23:3(665-685), 1993.
 * -# [Jang1997] J.-S.R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
 * -# [ISO2014] ISO, <em>ISO/IEC 14882-2014: Information technology - Programming languages - C++</em>, 2014
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API Engine: public fl::Engine
{
private:
    typedef fl::Engine BaseType;


public:
    /// The category for the layers of the ANFIS network
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


public:
    /**
     * Constructor
     *
     * \param name The mnemonic name for this engine
     */
    explicit Engine(const std::string& name = "");

    /**
     * Constructor
     *
     * \tparam InputIterT Iterator type to pointer to fuzzy input variables.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     * \tparam OutputIterT Iterator type to pointer to fuzzy output variables.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     * \tparam RuleBlockIterT Iterator type to pointer to fuzzy rule blocks.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     *
     * \param inputFirst The iterator to the beginning of the range of pointers to fuzzy input variables.
     * \param inputLast The iterator to the ending of the range of pointers to fuzzy input variables.
     * \param outputFirst The iterator to the beginning of the range of pointers to fuzzy output variables.
     * \param outputLast The iterator to the ending of the range of pointers to fuzzy output variables.
     * \param ruleBlockFirst The iterator to the beginning of the range of pointers to fuzzy rule blocks.
     * \param ruleBlockLast The iterator to the ending of the range of pointers to fuzzy rule blocks.
     * \param name The mnemonic name for this engine
     */
    template <typename InputIterT,
              typename OutputIterT,
              typename RuleBlockIterT>
    Engine(InputIterT inputFirst, InputIterT inputLast,
           OutputIterT outputFirst, OutputIterT outputLast,
           RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast,
           const std::string& name = "");

    /// Constructs a new object from an other object of base class
    explicit Engine(const fl::Engine& other);

    /// The copy constructor
    Engine(const Engine& other);

    FL_DEFAULT_MOVE(Engine)

    /// The destructor
    virtual ~Engine();

    /// The copy assignement
    Engine& operator=(const Engine& rhs);

    /// Clones this object
    Engine* clone() const;

    /**
     * Sets the input variables
     *
     * \tparam IterT Iterator type to pointer to input variables.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     *
     * \param first The iterator to the beginning of the range of pointer to input variables
     * \param last The iterator to the ending of the range of pointer to input variables
     */
    template <typename IterT>
    void setInputVariables(IterT first, IterT last);

    /**
     * Sets the output variables
     *
     * \tparam IterT Iterator type to pointer to output variables.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     *
     * \param first The iterator to the beginning of the range of pointer to output variables
     * \param last The iterator to the ending of the range of pointer to output variables
     */
    template <typename IterT>
    void setOutputVariables(IterT first, IterT last);

    /**
     * Sets the rule blocks
     *
     * \tparam IterT Iterator type to pointer to rule blocks.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     *
     * \param first The iterator to the beginning of the range of pointer to rule blocks.
     * \param last The iterator to the ending of the range of pointer to rule blocks.
     */
    template <typename IterT>
    void setRuleBlocks(IterT first, IterT last);

    /**
     * Tells if this engine is ready to perform inferences, and if not returns
     * a specific message in the \a status parameter
     *
     * \param status An optional output parameter used for setting the reason
     *  of why this engine is not ready to perform inferences.
     *
     * \return \c true if this engine is ready; \c false if not.
     */
    bool isReady(std::string* status = fl::null) const;

    /**
     * Performs fuzzy inference according to current settings, variables, and
     * rules.
     */
    void process();

    /// Resets this engine to perform a new inference
    void restart();

    /**
     * Builds the ANFIS model according to current settings, variables, and
     * rules
     */
    void build();

    /**
     * Builds the ANFIS model with the given parameters
     *
     * \tparam InputIterT Iterator type to pointer to fuzzy input variables.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     * \tparam OutputIterT Iterator type to pointer to fuzzy output variables.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     * \tparam RuleBlockIterT Iterator type to pointer to fuzzy rule blocks.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     *
     * \param inputFirst The iterator to the beginning of the range of pointers to fuzzy input variables.
     * \param inputLast The iterator to the ending of the range of pointers to fuzzy input variables.
     * \param outputFirst The iterator to the beginning of the range of pointers to fuzzy output variables.
     * \param outputLast The iterator to the ending of the range of pointers to fuzzy output variables.
     * \param ruleBlockFirst The iterator to the beginning of the range of pointers to fuzzy rule blocks.
     * \param ruleBlockLast The iterator to the ending of the range of pointers to fuzzy rule blocks.
     * \param name The mnemonic name for this engine
     */
    template <typename InputIterT,
              typename OutputIterT,
              typename RuleBlockIterT>
    void build(InputIterT inputFirst, InputIterT inputLast,
               OutputIterT outputFirst, OutputIterT outputLast,
               RuleBlockIterT ruleBlockFirst, RuleBlockIterT ruleBlockLast);

    /// Hard resets this engine, by removing all variables and rules
    void clear();

    /// Sets the flags indicating if this ANFIS is in learning mode
    void setIsLearning(bool value);

    /// Tells if this ANFIS is in learning mode
    bool isLearning() const;

    /// Sets the flags indicating if this ANFIS has a bias in the output nodes
    void setHasBias(bool value);

    /// Tells if this ANFIS has a bias in the output nodes
    bool hasBias() const;

    /**
     * Sets the bias values for the output nodes of this ANFIS
     *
     * \tparam IterT Iterator type to real numbers.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     *
     * \param first The iterator to the beginning of the range of real numbers representing bias values for the output nodes.
     * \param last The iterator to the ending of the range of real numbers representing bias values for the output nodes.
     */
    template <typename IterT>
    void setBias(IterT first, IterT last);

    /**
     * Sets the bias values for the output nodes of this ANFIS
     *
     * \param value The vector bias values for the output nodes.
     */
    void setBias(const std::vector<fl::scalar>& value);

    /// Gets the bias values of the output nodes of this ANFIS
    std::vector<fl::scalar> getBias() const;

    /// Returns the input connections to the given ANFIS node
    std::vector<Node*> inputConnections(const Node* p_node) const;

    /// Returns the output connections to the given ANFIS node
    std::vector<Node*> outputConnections(const Node* p_node) const;

    /**
     * Sets the values of the fuzzy input variables
     *
     * \tparam IterT Iterator type to real numbers.
     *  Must meet the requirements of \c InputIterator concept [ISO2014 sec. 24.2.3].
     *
     * \param first The iterator to the beginning of the range of real numbers representing values for the input nodes.
     * \param last The iterator to the ending of the range of real numbers representing bias values for the input nodes.
     */
    template <typename IterT>
    void setInputValues(IterT first, IterT last);

    /// Gets the current values of the input variables
    std::vector<fl::scalar> getInputValues() const;

    /// Gets the set of nodes in the input layer of the ANFIS network
    std::vector<InputNode*> getInputLayer() const;

    /// Gets the set of nodes in the fuzzification layer of the ANFIS network
    std::vector<FuzzificationNode*> getFuzzificationLayer() const;

    /// Gets the set of nodes in the input hedge layer of the ANFIS network
    std::vector<InputHedgeNode*> getInputHedgeLayer() const;

    /// Gets the set of nodes in the antecedent layer of the ANFIS network
    std::vector<AntecedentNode*> getAntecedentLayer() const;

    /// Gets the set of nodes in the consequent layer of the ANFIS network
    std::vector<ConsequentNode*> getConsequentLayer() const;

    /// Gets the set of nodes in the accumulation layer of the ANFIS network
    std::vector<AccumulationNode*> getAccumulationLayer() const;

    /// Gets the set of nodes in the output layer of the ANFIS network
    std::vector<OutputNode*> getOutputLayer() const;

    /// Gets the set of nodes of the given layer \a layer of the ANFIS network
    std::vector<Node*> getLayer(LayerCategory layer) const;

    /// Forwards the current input values to the ANFIS network and return the network outputs
    std::vector<fl::scalar> eval();

    /// Forwards the current input values to the ANFIS network until layer \a layer and returns the values of nodes of \a layer
    std::vector<fl::scalar> evalTo(LayerCategory layer);

    /// Forwards the current values of layer \a layer to the ANFIS network until the output layer and returns the network outputs
    std::vector<fl::scalar> evalFrom(LayerCategory layer);

    /// Forwards the input values in the range of iterators [\a first, \a last) to the ANFIS network until the output layer and return the network outputs
    template <typename IterT>
    std::vector<fl::scalar> eval(IterT first, IterT last);

    /// Forwards the input values in the range of iterators [\a first, \a last) to the ANFIS network until layer \a layer and return the values of nodes of \a layer
    template <typename IterT>
    std::vector<fl::scalar> evalTo(IterT first, IterT last, LayerCategory layer);

    /// Gets the layer category coming next to layer \a cat
    LayerCategory getNextLayerCategory(LayerCategory cat) const;

    /// Gets the layer category coming before of layer \a cat
    LayerCategory getPreviousLayerCategory(LayerCategory cat) const;

protected:
    /// Updates internal references and pointers
    void updateAnfisReferences();

private:
    /// Checks if this ANFIS model is well formed
    void check();

    /// Clears the ANFIS model structure (but not the fuzzy variables and rules)
    void clearAnfis();

    /// Connects the given nodes in this ANFIS model
    void connect(Node* p_from, Node* p_to);

    /// Evaluates the given layer \a layer and returns the values of the associated nodes
    std::vector<fl::scalar> evalLayer(LayerCategory layer);

    /// Evaluates the layer whose nodes are passed as parameters and returns the values of the nodes
    template <typename IterT>
    std::vector<fl::scalar> evalLayer(IterT first, IterT last);

    /// Evaluates the input layer and returns the values of the related nodes
    std::vector<fl::scalar> evalInputLayer();

    /// Evaluates the fuzzification layer and returns the values of the related nodes
    std::vector<fl::scalar> evalFuzzificationLayer();

    /// Evaluates the input hedge layer and returns the values of the related nodes
    std::vector<fl::scalar> evalInputHedgeLayer();

    /// Evaluates the antecedent layer and returns the values of the related nodes
    std::vector<fl::scalar> evalAntecedentLayer();

    /// Evaluates the consequent layer and returns the values of the related nodes
    std::vector<fl::scalar> evalConsequentLayer();

    /// Evaluates the accumulation layer and returns the values of the related nodes
    std::vector<fl::scalar> evalAccumulationLayer();

    /// Evaluates the output layer and returns the values of the related nodes
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


////////////////////////
// Template definitions
////////////////////////


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
    //  outputNodes_[i]->setBias(0);
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


#endif // FL_ANFIS_ENGINE_H
