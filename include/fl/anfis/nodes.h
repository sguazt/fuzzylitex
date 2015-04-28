/**
 * \file fl/anfis/nodes.h
 *
 * \brief Nodes of the adaptive network underlying an ANFIS model
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

#ifndef FL_ANFIS_NODES_H
#define FL_ANFIS_NODES_H


#include <boost/noncopyable.hpp>
//#include <fl/anfis/engine.h>
#include <fl/fuzzylite.h>
#include <fl/hedge/Hedge.h>
#include <fl/norm/Norm.h>
//#include <fl/norm/TNorm.h>
//#include <fl/norm/SNorm.h>
#include <fl/term/Term.h>
#include <fl/variable/InputVariable.h>
#include <fl/variable/OutputVariable.h>
#include <vector>


namespace fl { namespace anfis {

class Engine;

class Node//: boost::noncopyable
{
private:
	FL_DISABLE_COPY(Node)

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
	ConsequentNode(fl::Term* p_term, /*fl::TNorm* p_tnorm,*/ Engine* p_engine);

	fl::Term* getTerm() const;

//	fl::TNorm* getTNorm() const;

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();


private:
	fl::Term* p_term_;
//	fl::TNorm* p_tnorm_;
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
	OutputNode(fl::OutputVariable* p_var, Engine* p_engine);

	fl::OutputVariable* getOutputVariable() const;

	void setBias(fl::scalar value);

	fl::scalar getBias() const;

private:
	fl::scalar doEval();

	std::vector<fl::scalar> doEvalDerivativeWrtInputs();


private:
	fl::OutputVariable* p_var_;
	fl::scalar bias_; /// The value to use in place of the output value in case of zero firing strength
}; // OutputNode

}} // Namespace fl::anfis

#endif // FL_ANFIS_NODES_H
