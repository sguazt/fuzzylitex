/*
 * Copyright (c) 1991 Jyh-Shing Roger Jang, Dept of EECS, U.C. Berkeley
 * BISC (Berkeley Initiative on Soft Computing) group
 * jang@eecs.berkeley.edu
 * Permission is granted to modify and re-distribute this code in any manner
 * as long as this notice is preserved.  All standard disclaimers apply.
 */

#ifndef FL_ANFIS_ENGINE_HPP
#define FL_ANFIS_ENGINE_HPP


#include <cstddef>
#include <fl/Headers.h>
#include <map>
#include <my/ann.hpp>
#include <my/Commons.hpp>
#include <stdexcept>
#include <vector>


namespace fl {

enum FisType
{
	HybridFis = fl::Engine::Hybrid,
	InverseTsukamotoFis = fl::Engine::InverseTsukamoto,
	LarsenFis = fl::Engine::Larsen,
	MamdaniFis = fl::Engine::Mamdani,
	TakagiSugenoFis = fl::Engine::TakagiSugeno,
	TsukamotoFis = fl::Engine::Tsukamoto,
	UnknownFis = fl::Engine::Unknown
};

//FIXME: maybe to be moved inside the Operation class?
template <typename EngineT>
FisType fisType(const EngineT& eng)
{
	const std::vector<OutputVariable*> outputs = eng.outputVariables();

	if  (outputs.empty())
	{
		return UnknownFis;
	}

	// Mamdani FIS
	bool mamdani = true;
	for (std::size_t i = 0,
					 n = outputs.size();
		 i < n && mamdani;
		 ++i)
	{
		const OutputVariable* p_output = outputs[i];

		//Defuzzifier must be integral
		mamdani &= static_cast<bool>(dynamic_cast<IntegralDefuzzifier*>(p_output->getDefuzzifier()));
	}

	// Larsen FIS
	// NOTE: Larsen FIS is a Mamdani FIS with algebraic-product as activation operator
	bool larsen = mamdani;
	if (larsen)
	{
		const std::vector<RuleBlock*> ruleBlocks = eng.ruleBlocks();
		larsen &= !ruleBlocks.empty();
		if (larsen)
		{
			for (std::size_t i = 0,
							 n = ruleBlocks.size();
				 i < n;
				 ++i)
			{
				const RuleBlock* p_ruleBlock = ruleBlocks[i];

				larsen &= static_cast<bool>(dynamic_cast<const AlgebraicProduct*>(p_ruleBlock->getActivation()));
			}
		}
		if (larsen)
		{
			return LarsenFis;
		}
	}
	if (mamdani)
	{
		return MamdaniFis;
	}
	//Else, keep checking

	// Takagi-Sugeno FIS
	bool sugeno = true;
	for (std::size_t i = 0,
					 no = outputs.size();
		 i < no;
		 ++i)
	{
		const OutputVariable* p_output = outputs[i];

		// Defuzzifier is Weighted
		const WeightedDefuzzifier* p_weightedDefuzzifier = dynamic_cast<WeightedDefuzzifier*>(p_output->getDefuzzifier());

		sugeno &= static_cast<bool>(p_weightedDefuzzifier)
				  && (p_weightedDefuzzifier->getType() == WeightedDefuzzifier::Automatic
					  || p_weightedDefuzzifier->getType() == WeightedDefuzzifier::TakagiSugeno);

		if (sugeno)
		{
			// Takagi-Sugeno has only Constant, Linear or Function terms
			for (std::size_t t = 0,
							 nt = p_output->numberOfTerms();
				 t < nt && sugeno;
				 ++t)
			{
				const Term* p_term = p_output->getTerm(t);

				sugeno &= p_weightedDefuzzifier->inferType(p_term) == WeightedDefuzzifier::TakagiSugeno;
			}
		}
	}
	if (sugeno)
	{
		return TakagiSugenoFis;
	}

	// Tsukamoto FIS
	bool tsukamoto = true;
	for (std::size_t i = 0,
					 no = outputs.size();
		 i < no && tsukamoto;
		 ++i)
	{
		const OutputVariable* p_output = outputs[i];

		// Defuzzifier is Weighted
		const WeightedDefuzzifier* p_weightedDefuzzifier = dynamic_cast<WeightedDefuzzifier*>(p_output->getDefuzzifier());

		tsukamoto &= static_cast<bool>(p_weightedDefuzzifier)
					 && (p_weightedDefuzzifier->getType() == WeightedDefuzzifier::Automatic
						 || p_weightedDefuzzifier->getType() == WeightedDefuzzifier::Tsukamoto);
		if (tsukamoto)
		{
			// Tsukamoto has only monotonic terms: Concave, Ramp, Sigmoid, SShape, or ZShape
			for (std::size_t t = 0,
							 nt = p_output->numberOfTerms();
				 t < nt && tsukamoto;
				 ++t)
			{
				const Term* p_term = p_output->getTerm(t);
				tsukamoto &= p_weightedDefuzzifier->isMonotonic(p_term);
			}
		}
	}
	if (tsukamoto)
	{
		return TsukamotoFis;
	}

	// Inverse Tsukamoto FIS
	bool inverseTsukamoto = true;
	for (std::size_t i = 0,
					 no = outputs.size();
		 i < no && inverseTsukamoto;
		 ++i)
	{
		const OutputVariable* p_output = outputs[i];

		// Defuzzifier cannot be integral
		const WeightedDefuzzifier* p_weightedDefuzzifier = dynamic_cast<WeightedDefuzzifier*>(p_output->getDefuzzifier());
		inverseTsukamoto &= static_cast<bool>(p_weightedDefuzzifier);
	}
	if (inverseTsukamoto)
	{
		return InverseTsukamotoFis;
	}

	bool hybrid = true;
	for (std::size_t i = 0,
					 no = outputs.size();
		 i < no;
		 ++i)
	{
		const OutputVariable* p_output = outputs[i];

		// Output variables have non-fl::null defuzzifiers
		hybrid &= static_cast<bool>(p_output->getDefuzzifier());
	}
	if (hybrid)
	{
		return HybridFis;
	}

	return UnknownFis;
}

/*
class io_node
{
	private: std::string name_;
	private: fl::scalar val_;
	private: fl::scalar lb_;
	private: fl::scalar ub_;
	private: std::vector< boost::shared_ptr< mf_node<fl::scalar> > > mf_nodes_;
}; // io_node

class fis_node
{
}; // fis_node
*/

namespace detail {

template <typename EngineT>
void check_fis_engine_v5_concept()
{
	EngineT eng;

	eng.setName("name");

	eng.addInputVariable(new InputVariable("in1"));

	eng.addOutputVariable(new OutputVariable("out1"));

	eng.addRuleBlock(new RuleBlock());

	const std::string name = eng.getName();

	const std::vector<InputVariable*> inputs = eng.inputVariables();

	const InputVariable* p_in1 = eng.getInputVariable(0);
	const InputVariable* p_in2 = eng.getInputVariable("in1");

	const std::vector<OutputVariable*> outputs = eng.outputVariables();

	const OutputVariable* p_out1 = eng.getOutputVariable(0);
	const OutputVariable* p_out2 = eng.getOutputVariable("out1");

	const std::vector<RuleBlock*> ruleBlocks = eng.ruleBlocks();

	const std::vector<Variable*> variables = eng.variables();

	const FisType type = eng.type();
	std::string typeName;
	std::string typeReason;
	const FisType type2 = eng.type(&typeName, &typeReason);

	try { eng.restart(); } catch (...) { }

	try { eng.setInputValue("in1", nan); } catch (...) { }

	try { eng.process(); } catch (...) { }

	try { scalar output = eng.getOutputValue("out1"); } catch (...) { }

	EngineT* cloned = eng.clone();

	eng.toString();
}

} // Namespace detail


namespace anfis {

class FL_API Engine/*: public fl::Engine*/
{
	public: explicit Engine(const std::string& name = "")
	: name_(name)/*,
	  p_conj_(fl::null),
	  p_disj_(fl::null),
	  p_activ_(fl::null),
	  p_accum_(fl::null),
	  p_defuz_(fl::null)*/
	{
	}

	public: Engine(const Engine& other)
	{
		//TODO
		FL_THROW2(std::runtime_error, "To be implemented");
	}

	public: virtual ~Engine()
	{
		this->reset();
	}

	//TODO
	//FL_DEFAULT_MOVE(Engine)

	public: Engine& operator=(const Engine& rhs)
	{
		//TODO
		FL_THROW2(std::runtime_error, "To be implemented");
		return *this;
	}

	public: Engine* clone() const
	{
		//FIXME: Possible source of mem leak. Use smart-pointer, instead!
		return new Engine(*this);
	}

	public: void setName(std::string const& value)
	{
		name_ = value;
	}

	public: std::string getName() const
	{
		return name_;
	}

/*FIXME: these methods seem redundant as this info is set inside input variables, output variables and ruleblocks
	public: void configure(const std::string& conjunctionT,
						   const std::string& disjunctionS,
						   const std::string& activationT,
						   const std::string& accumulationS,
						   const std::string& defuzzifier,
						   int resolution = IntegralDefuzzifier::defaultResolution())
	{
		this->setConjunctionOperator(conjunctionT);
		this->setDisjunctionOperator(disjunctionS);
		this->setActivationOperator(activationT);
		this->setAccumulationOperator(accumulationS);
		this->setDefuzzifierOperator(defuzzifier, resolution);
	}

	public: void configure(TNorm* conjunctionT,
						   SNorm* disjunctionS,
						   TNorm* activationT,
						   SNorm* accumulationS,
						   Defuzzifier* defuzzifier)
	{
		this->setConjunctionOperator(conjunctionT);
		this->setDisjunctionOperator(disjunctionS);
		this->setActivationOperator(activationT);
		this->setAccumulationOperator(accumulationS);
		this->setDefuzzifierOperator(defuzzifier);
	}

	public: void setConjunctionOperator(TNorm* p_conj)
	{
		if (!p_conj)
		{
			FL_THROW("Invalid argument: conjunction operator not specified");
		}

		p_conj_ = p_conj;
	}

	public: void setConjunctionOperator(const std::string& name)
	{
		p_conj_ = FactoryManager::instance()->tnorm()->constructObject(name);

		if (!p_conj_)
		{
			FL_THROW("Invalid argument: unknown conjunction operator");
		}
	}

	public: TNorm* getConjunctionOperator() const
	{
		return p_conj_;
	}

	public: void setDisjunctionOperator(SNorm* p_disj)
	{
		if (!p_disj)
		{
			FL_THROW("Invalid argument: disjunction operator not specified");
		}

		p_disj_ = p_disj;
	}

	public: void setDisjunctionOperator(const std::string& name)
	{
		p_disj_ = FactoryManager::instance()->snorm()->constructObject(name);

		if (!p_disj_)
		{
			FL_THROW("Invalid argument: unknown disjunction operator");
		}
	}

	public: SNorm* getDisjunctionOperator() const
	{
		return p_disj_;
	}

	public: void setActivationOperator(TNorm* p_activ)
	{
		if (!p_activ)
		{
			FL_THROW("Invalid argument: activation operator not specified");
		}

		p_activ_ = p_activ;
	}

	public: void setActivationOperator(const std::string& name)
	{
		p_activ_ = FactoryManager::instance()->tnorm()->constructObject(name);

		if (!p_disj_)
		{
			FL_THROW("Invalid argument: unknown activation operator");
		}
	}

	public: TNorm* getActivationOperator() const
	{
		return p_activ_;
	}

	public: void setAccumulationOperator(SNorm* p_accum)
	{
		if (!p_accum)
		{
			FL_THROW("Invalid argument: accumulation operator not specified");
		}

		p_accum_ = p_accum;
	}

	public: void setAccumulationOperator(const std::string& name)
	{
		p_accum_ = FactoryManager::instance()->snorm()->constructObject(name);

		if (!p_accum_)
		{
			FL_THROW("Invalid argument: unknown accumulation operator");
		}
	}

	public: SNorm* getAccumulationOperator() const
	{
		return p_accum_;
	}

	public: void setDefuzzifierOperator(Defuzzifier* p_defuz)
	{
		p_defuz_ = p_defuz;
	}

	public: void setDefuzzifierOperator(const std::string& name,
										int resolution = IntegralDefuzzifier::defaultResolution())
	{
		p_defuz_ = FactoryManager::instance()->defuzzifier()->constructObject(name);

		IntegralDefuzzifier* p_intDefuz = dynamic_cast<IntegralDefuzzifier*>(p_defuz_);
		if (p_intDefuz)
		{
			p_intDefuz->setResolution(resolution);
		}
	}

	public: Defuzzifier* getDefuzzifierOperator() const
	{
		return p_defuz_;
	}
*/

	public: void addInputVariable(InputVariable* var)
	{
		inputs_.push_back(var);
	}

	public: InputVariable* getInputVariable(std::size_t idx) const
	{
		if (idx >= inputs_.size())
		{
			FL_THROW2(std::out_of_range, "Index to input variable is out of range");
		}

		return this->inputs_[idx];
    }

    public: InputVariable* getInputVariable(const std::string& name) const
	{
		for (std::size_t i = 0,
						 n = inputs_.size();
			 i < n;
			 ++i)
		{
			InputVariable* p_input = inputs_[i];

			//check: null
			FL_DEBUG_ASSERT( p_input );

			if (p_input->getName() == name)
			{
				return p_input;
			}
		}

		FL_THROW("[engine error] input variable <" + name + "> not found");
    }

	public: std::vector<InputVariable*> inputVariables() const
	{
		return inputs_;
	}

	public: void addOutputVariable(OutputVariable* var)
	{
		outputs_.push_back(var);
	}

	public: OutputVariable* getOutputVariable(std::size_t idx) const
	{
		if (idx >= outputs_.size())
		{
			FL_THROW2(std::out_of_range, "Index to output variable is out of range");
		}

		return this->outputs_[idx];
    }

    public: OutputVariable* getOutputVariable(const std::string& name) const
	{
		for (std::size_t i = 0,
						 n = outputs_.size();
			 i < n;
			 ++i)
		{
			OutputVariable* p_output = outputs_[i];

			//check: null
			FL_DEBUG_ASSERT( p_output );

			if (p_output->getName() == name)
			{
				return p_output;
			}
		}

		FL_THROW("[engine error] output variable <" + name + "> not found");
    }

	public: std::vector<OutputVariable*> outputVariables() const
	{
		return outputs_;
	}

	public: std::vector<Variable*> variables() const
	{
		std::vector<Variable*> vars(inputs_.size()+outputs_.size());

		vars.assign(inputs_.begin(), inputs_.end());
		vars.insert(vars.end(), outputs_.begin(), outputs_.end());

		return vars;
	}

	public: void addRuleBlock(RuleBlock* p_block)
	{
		//FIXME: what does it happens if multiple rule blocks are enabled and share the same output variables?

		if (!p_block)
		{
			FL_THROW("Invalid argument: rule block not specified");
		}

		ruleBlocks_.push_back(p_block);
	}

	public: std::vector<RuleBlock*> ruleBlocks() const
	{
		return ruleBlocks_;
	}

	//FIXME: what's that?
	/// Sets the bias to handle zero firing error
	public: template <typename IterT>
			void setBias(IterT first, IterT last)
	{
		bias_.assign(first, last);
	}

	/// Gets the bias to handle zero firing error
	public: std::vector<fl::scalar> getBias() const
	{
		return bias_;
	}

	public: void setInputValue(const std::string& name, scalar value)
	{
		InputVariable* p_input = this->getInputVariable(name);
		p_input->setInputValue(value);
	}

	public: scalar getOutputValue(const std::string& name) const
	{
		OutputVariable* p_output = this->getOutputVariable(name);
		return p_output->getOutputValue();
	}

	public: FisType type(std::string* name = null, std::string* reason = null) const
	{
		const FisType res = fisType(*this);

		switch (res)
		{
			case HybridFis:
				if (name) *name = "Hybrid";
				if (reason) *reason = "- Output variables have different defuzzifiers";
				break;
			case InverseTsukamotoFis:
				if (name) *name = "Inverse Tsukamoto";
				if (reason) *reason = "- Output variables have weighted defuzzifiers\n"
									  "- Output variables do not only have constant, linear or function terms\n"
									  "- Output variables do not only have monotonic terms\n";
				break;
			case LarsenFis:
				if (name) *name = "Larsen";
				if (reason) *reason = "- Output variables have integral defuzzifiers\n"
									  "- Rule blocks activate using the algebraic product T-Norm";
				break;
			case MamdaniFis:
				if (name) *name = "Mamdani";
				if (reason) *reason = "-Output variables have integral defuzzifiers";
				break;
			case TakagiSugenoFis:
				if (name) *name = "Takagi-Sugeno";
				if (reason) *reason = "- Output variables have weighted defuzzifiers\n"
									  "- Output variables have constant, linear or function terms";
				break;
			case TsukamotoFis:
				if (name) *name = "Tsukamoto";
				if (reason) *reason = "- Output variables have weighted defuzzifiers\n"
									  "- Output variables only have monotonic terms";
				break;
			case UnknownFis:
				if (name) *name = "Unknown";
				if (reason) *reason = (outputs_.size() == 0)
									  ? "- Engine has no output variables"
									  : "- There are output variables without a defuzzifier";
				break;
		}

		return res;
	}

	public: void learn()
	{
		this->checkForLearning();
		this->build();
		FL_THROW2(std::runtime_error, "To be implemented");
	}

	public: void reset()
	{
        for (std::size_t i = 0,
						 n = ruleBlocks_.size();
			 i < n;
			 ++i)
		{
			delete ruleBlocks_.at(i);
		}
        for (std::size_t i = 0,
						 n = inputs_.size();
			 i < n;
			 ++i)
		{
			delete inputs_.at(i);
		}
        for (std::size_t i = 0,
						 n = outputs_.size();
			 i < n;
			 ++i)
		{
			delete outputs_.at(i);
		}
//		if (p_conj_)
//		{
//			delete p_conj_;
//		}
//		if (p_disj_)
//		{
//			delete p_disj_;
//		}
//		if (p_activ_)
//		{
//			delete p_activ_;
//		}
//		if (p_accum_)
//		{
//			delete p_accum_;
//		}
//		if (p_defuz_)
//		{
//			delete p_defuz_;
//		}
	}

	public: void restart()
	{
		//TODO
		FL_THROW2(std::runtime_error, "To be implemented");
	}

	public: void process()
	{
		//TODO
		FL_THROW2(std::runtime_error, "To be implemented");
	}

	public: std::string toString() const
	{
		//TODO
		//return FllExporter().toString(this);
		FL_THROW2(std::runtime_error, "To be implemented");
		return "";
	}

	private: void checkForLearning() const
	{
		if (this->type() != TakagiSugenoFis)
		{
			FL_THROW("ANFIS can only be used with Takagi-Sugeno FIS");
		}
		if (outputs_.size() > 1)
		{
			FL_THROW("ANFIS can only have one output variable");
		}
		if (!dynamic_cast<const WeightedAverage*>(outputs_.back()->getDefuzzifier()))
		{
			FL_THROW("ANFIS can only use the weighted-average method as defuzzifier operator");
		}
		std::size_t numEnabled = 0;
		std::size_t enabledIdx = 0;
		for (std::size_t i = 0,
						 nrb = ruleBlocks_.size();
			 i < nrb && numEnabled < 2;
			 ++i)
		{
			const RuleBlock* p_ruleBlock = ruleBlocks_[i];

			// check: null
			FL_DEBUG_ASSERT( p_ruleBlock );

			if (p_ruleBlock->isEnabled())
			{
				++numEnabled;
				enabledIdx = i;
			}
		}
		if (numEnabled == 0)
		{
			FL_THROW("ANFIS must have one rule block enabled");
		}
		else if (numEnabled > 1)
		{
			FL_THROW("ANFIS can only have one rule block enabled at a time");
		}

		const RuleBlock* p_ruleBlock = ruleBlocks_[enabledIdx];

		// check: null
		FL_DEBUG_ASSERT( p_ruleBlock );

		const OutputVariable* p_output = outputs_.back();

		// check: null
		FL_DEBUG_ASSERT( p_output );

		if (p_ruleBlock->numberOfRules() <= 1)
		{
			FL_THROW("ANFIS needs at least two rules");
		}

		// Check for parameter sharing
		if (p_output->numberOfTerms() != p_ruleBlock->numberOfRules())
		{
			FL_THROW("In ANFIS, the number of output terms must be the same as the number of rules (output parameter sharing is not allowed)");
		}
		else
		{
			// Check that each output term is used exaclty once
			for (std::size_t t = 0,
							 nt = p_output->numberOfTerms();
				 t < nt;
				 ++t)
			{
				const Term* p_term = p_output->getTerm(t);

				// check: null
				FL_DEBUG_ASSERT( p_term );

				const std::string termName = p_term->getName();

				std::size_t count = 0;
				for (std::size_t r = 0,
								 nr = p_ruleBlock->numberOfRules();
					 r < nr && count < 2;
					 ++r)
				{
					const Rule* p_rule = p_ruleBlock->getRule(r);

					// check: null
					FL_DEBUG_ASSERT( p_rule );

					const Consequent* p_cons = p_rule->getConsequent();

					// check: null
					FL_DEBUG_ASSERT( p_cons );

					const std::vector<Proposition*> props = p_cons->conclusions();

					for (std::size_t p = 0,
									 np = props.size();
						 p < np && count < 2;
						 ++p)
					{
						const Proposition* p_prop = props[p];

						// check: null
						FL_DEBUG_ASSERT( p_prop );

						// check: null
						FL_DEBUG_ASSERT( p_prop->term );

						if (p_prop->term->getName() == termName)
						{
							++count;
						}
					}
				}

				if (count > 1)
				{
					FL_THROW("In ANFIS, an output term can be used only once (output parameter sharing is not allowed)");
				}
			}
		}

		// All output MFs must be linear or constant
		bool allConstTerms = true;
		bool allLinTerms = true;
		for (std::size_t t = 0,
						 nt = p_output->numberOfTerms();
			 t < nt && (allConstTerms || allLinTerms);
			 ++t)
		{
			const Term* p_term = p_output->getTerm(t);

			// check: null
			FL_DEBUG_ASSERT( p_term );

			allConstTerms &= static_cast<bool>(dynamic_cast<Constant const*>(p_term));
			allLinTerms &= static_cast<bool>(dynamic_cast<Linear const*>(p_term));
		}
		if (!(allConstTerms ^ allLinTerms))
		{
			FL_THROW("In ANFIS, each output MF must be either all constant or all linear");
		}

		// All rule weights must be one
		for (std::size_t r = 0,
						 nr = p_ruleBlock->numberOfRules();
			 r < nr;
			 ++r)
		{
			const Rule* p_rule = p_ruleBlock->getRule(r);

			// check: null
			FL_DEBUG_ASSERT( p_rule );

			if (Operation::isGt(p_rule->getWeight(), 1))
			{
				FL_THROW("In ANFIS, the weight of each rule must be exactly one");
			}
		}
	}

	private: void build()
	{
		OutputVariable* p_output = outputs_.back();

		FL_DEBUG_ASSERT( p_output );

		if (p_output->numberOfTerms() > 0)
		{
			const Term* p_term = p_output->getTerm(0);
			if (dynamic_cast<Constant const*>(p_term))
			{
				order_ = 0;
			}
			else if (dynamic_cast<Linear const*>(p_term))
			{
				order_ = 1;
			}
		}

		// Build the neural network

		ann::Layer<scalar>* p_layer;

		nnet_.reset();

		std::map<Variable*,ann::Neuron<scalar>*> varNodeMap;

		// Layer 0 (the input layer): one node for each input variable
		p_layer = new ann::Layer<scalar>();
		for (std::size_t i = 0,
						 n = inputs_.size();
			 i < n;
			 ++i)
		{
			ann::Neuron<scalar>* p_neuron = new ann::Neuron<scalar>();
			p_layer->addNeuron(p_neuron);
			varNodeMap[inputs_[i]] = p_neuron;
		}
		nnet_.addLayer(p_layer);

		// Layer 1: one node for each term of each input variable, where the activation function is the term's MF
		p_layer = new ann::Layer<scalar>();
		for (std::size_t i = 0,
						 ni = inputs_.size();
			 i < ni;
			 ++i)
		{
			for (std::size_t t = 0,
							 nt = inputs_[i]->numberOfTerms();
				 t < nt;
				 ++t)
			{
				ann::Neuron<scalar>* p_neuron = new ann::Neuron<scalar>();
				p_layer->addNeuron(p_neuron);
			}
		}
		nnet_.addLayer(p_layer);

		// Layer 2: one node for each term of each input variable, where the activation function is the conjunction operator
		p_layer = new ann::Layer<scalar>();
		for (std::size_t i = 0,
						 ni = inputs_.size();
			 i < ni;
			 ++i)
		{
			for (std::size_t t = 0,
							 nt = inputs_[i]->numberOfTerms();
				 t < nt;
				 ++t)
			{
				ann::Neuron<scalar>* p_neuron = new ann::Neuron<scalar>();
				p_layer->addNeuron(p_neuron);
			}
		}
		nnet_.addLayer(p_layer);

		// Layer 3: one node for each rule, where the activation function is the normalized firing strength
		p_layer = new ann::Layer<scalar>();
		for (std::size_t b = 0,
						 nb = ruleBlocks_.size();
			 b < nb;
			 ++b)
		{
			const RuleBlock* p_ruleBlock = ruleBlocks_[b];

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
				ann::Neuron<scalar>* p_neuron = new ann::Neuron<scalar>();
				p_layer->addNeuron(p_neuron);
			}
		}
		nnet_.addLayer(p_layer);

		// Layer 4: one node for each rule, where the activation function is the rule's consequent weighted by the normalized firing strength
		p_layer = new ann::Layer<scalar>();
		for (std::size_t b = 0,
						 nb = ruleBlocks_.size();
			 b < nb;
			 ++b)
		{
			const RuleBlock* p_ruleBlock = ruleBlocks_[b];

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
				ann::Neuron<scalar>* p_neuron = new ann::Neuron<scalar>();
				p_layer->addNeuron(p_neuron);
			}
		}
		nnet_.addLayer(p_layer);

		// Layer 5: two nodes only,
		p_layer = new ann::Layer<scalar>();
		p_layer->addNeuron(new ann::Neuron<scalar>());
		p_layer->addNeuron(new ann::Neuron<scalar>());
		nnet_.addLayer(p_layer);

		// Layer 6: one node only
		p_layer = new ann::Layer<scalar>();
		{
			ann::Neuron<scalar>* p_neuron = new ann::Neuron<scalar>();
			p_layer->addNeuron(p_neuron);
			varNodeMap[outputs_[0]] = p_neuron;
		}
		nnet_.addLayer(p_layer);
	}


	private: std::string name_; ///< The mnemonic name for this FIS engine
	private: std::vector<InputVariable*> inputs_; ///< Collection of (pointer to) input variables
	private: std::vector<OutputVariable*> outputs_; ///< Collection of (pointer to) output variables
	private: std::vector<RuleBlock*> ruleBlocks_; ///< Collection of (pointer to) rule blocks
//	private: TNorm* p_conj_; ///< Pointer to the conjunction (or AND) operator to compute the firing strength of a rule with AND'ed antecedents
//	private: SNorm* p_disj_; ///< Pointer to the disjunction (or OR) operator to compute the firing strength of a rule with OR'ed antecedents
//	private: TNorm* p_activ_; ///< Pointer to the activation (or implication) operator to compute qualified consequent MFs based on given firing strength
//	private: SNorm* p_accum_; ///< Pointer to the accumulation (or aggregation) operator to aggregate qualified consequent MFs to generate an overall output MF
//	private: Defuzzifier* p_defuz_; ///< Pointer to the defuzzifier operator to transform an output MF to a crisp single output value
	//private: fis_category type_; ///< The type of the fuzzy inference system
	private: std::vector<fl::scalar> bias_; //FIXME:to be moved inside NN?
	private: std::size_t order_;
	private: fl::ann::NeuralNetwork<scalar> nnet_;
}; // Engine

/*
anfis make_anfis(std::vector< std::vector<fl::scalar> > const& train_data,
						fis_category cat,
						)
{
	FL_ASSERT(cat == takagi_sugeno);

	if (cat != takagi_sugeno)
	{
		FL_THROW("Only Takagi-Sugeno fuzzy inference system is supported by ANFIS");
	}

	FL_shared_ptr< fid_node<RealT> > p_fis(new fis_node<RealT>());
	p_fis->category(cat);


	return ret;
}
*/

}} // Namespace fl::anfis

#endif // FL_ANFIS_ENGINE_HPP
