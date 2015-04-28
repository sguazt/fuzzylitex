#include <algorithm>
#include <cmath>
#include <fl/anfis.h>
#include <fl/fuzzylite.h>
#include <fl/Headers.h>
#include <iostream>
#include <stdexcept>
#include <vector>


namespace /*<unnnamed>*/ { namespace detail {

void SetupSisoSugenoEngine(fl::Engine* p_eng)
{
	p_eng->setName("Simple Dimmer");

	fl::InputVariable* p_iv = new fl::InputVariable();
	p_iv->setEnabled(true);
	p_iv->setName("Ambient");
	p_iv->setRange(0.000, 1.000);
	p_iv->addTerm(new fl::Triangle("DARK", 0.000, 0.250, 0.500));
	p_iv->addTerm(new fl::Triangle("MEDIUM", 0.250, 0.500, 0.750));
	p_iv->addTerm(new fl::Triangle("BRIGHT", 0.500, 0.750, 1.000));
	p_eng->addInputVariable(p_iv);

	fl::OutputVariable* p_ov = new fl::OutputVariable();
	p_ov->setEnabled(true);
	p_ov->setName("Power");
	p_ov->setRange(0.000, 1.000);
	p_ov->fuzzyOutput()->setAccumulation(fl::null);
	p_ov->setDefuzzifier(new fl::WeightedAverage());
	p_ov->setDefaultValue(fl::nan);
	p_ov->setLockPreviousValue(false);
	p_ov->setLockValueInRange(false);
	p_ov->addTerm(new fl::Constant("LOW", 0.250));
	p_ov->addTerm(new fl::Constant("MEDIUM", 0.500));
	p_ov->addTerm(new fl::Constant("HIGH", 0.750));
	p_eng->addOutputVariable(p_ov);

	fl::RuleBlock* p_rules = new fl::RuleBlock();
	p_rules->setEnabled(true);
	p_rules->setName("");
	p_rules->setActivation(fl::null);
	p_rules->setConjunction(new fl::AlgebraicProduct());
	p_rules->setDisjunction(new fl::AlgebraicSum());
	p_rules->addRule(fl::Rule::parse("if Ambient is DARK then Power is HIGH", p_eng));
	p_rules->addRule(fl::Rule::parse("if Ambient is MEDIUM then Power is MEDIUM", p_eng));
	p_rules->addRule(fl::Rule::parse("if Ambient is BRIGHT then Power is LOW", p_eng));
	p_eng->addRuleBlock(p_rules);
}

void SetupMisoSugenoEngine(fl::Engine* p_eng)
{
	p_eng->setName("Tipper (Sugeno)");

	fl::InputVariable* p_iv = 0;

	p_iv = new fl::InputVariable();
	p_iv->setEnabled(true);
	p_iv->setName("SERVICE");
	p_iv->setRange(0, 10);
	p_iv->addTerm(new fl::Gaussian("POOR", 0, 1.5));
	p_iv->addTerm(new fl::Gaussian("AVERAGE", 5, 1.5));
	p_iv->addTerm(new fl::Gaussian("GOOD", 10, 1.5));
	p_eng->addInputVariable(p_iv);

	p_iv = new fl::InputVariable();
	p_iv->setEnabled(true);
	p_iv->setName("FOOD");
	p_iv->setRange(0, 10);
	p_iv->addTerm(new fl::Trapezoid("RANCID", -5, 0, 1, 3));
	p_iv->addTerm(new fl::Trapezoid("DELICIOUS", 7, 9, 10, 15));
	p_eng->addInputVariable(p_iv);

	std::vector<fl::scalar> params;

	fl::OutputVariable* p_ov = 0;
	p_ov = new fl::OutputVariable();
	p_ov->setEnabled(true);
	p_ov->setName("TIP");
	p_ov->setRange(-30, 30);
	p_ov->fuzzyOutput()->setAccumulation(fl::null);
	p_ov->setDefuzzifier(new fl::WeightedAverage());
	p_ov->setDefaultValue(fl::nan);
	p_ov->setPreviousValue(false);
	params.clear(); params.push_back(0); params.push_back(0); params.push_back(5);
	p_ov->addTerm(new fl::Linear("CHEAP", params, p_eng));
	params.clear(); params.push_back(0); params.push_back(0); params.push_back(15);
	p_ov->addTerm(new fl::Linear("AVERAGE", params, p_eng));
	params.clear(); params.push_back(0); params.push_back(0); params.push_back(25);
	p_ov->addTerm(new fl::Linear("GENEROUS", params, p_eng));
	p_eng->addOutputVariable(p_ov);

	fl::RuleBlock* p_rules = new fl::RuleBlock();
	p_rules->setEnabled(true);
	p_rules->setConjunction(new fl::AlgebraicProduct());
	p_rules->setDisjunction(new fl::AlgebraicSum());
	p_rules->setActivation(fl::null);
	p_rules->addRule(fl::Rule::parse("if SERVICE is POOR or FOOD is RANCID then TIP is CHEAP", p_eng));
	p_rules->addRule(fl::Rule::parse("if SERVICE is AVERAGE then TIP is AVERAGE", p_eng));
	p_rules->addRule(fl::Rule::parse("if SERVICE is GOOD or FOOD is DELICIOUS then TIP is GENEROUS", p_eng));
	p_eng->addRuleBlock(p_rules);
}

void SetupMimoSugenoEngine(fl::Engine* p_eng)
{
	p_eng->setName("MIMO (Sugeno)");

	fl::InputVariable* p_iv = 0;

	p_iv = new fl::InputVariable();
	p_iv->setEnabled(true);
	p_iv->setName("X1");
	p_iv->setRange(0, 10);
	p_iv->addTerm(new fl::Gaussian("A1", 0, 1.5));
	p_iv->addTerm(new fl::Gaussian("A2", 5, 1.5));
	p_iv->addTerm(new fl::Gaussian("A3", 10, 1.5));
	p_eng->addInputVariable(p_iv);

	p_iv = new fl::InputVariable();
	p_iv->setEnabled(true);
	p_iv->setName("X2");
	p_iv->setRange(0, 10);
	p_iv->addTerm(new fl::Triangle("B1", 0, 0.5, 1));
	p_iv->addTerm(new fl::Triangle("B2", 0.5, 1, 1.5));
	p_eng->addInputVariable(p_iv);

	fl::OutputVariable* p_ov = 0;

	p_ov = new fl::OutputVariable();
	p_ov->setEnabled(true);
	p_ov->setName("Y1");
	p_ov->setRange(0, 30);
	p_ov->fuzzyOutput()->setAccumulation(fl::null);
	p_ov->setDefuzzifier(new fl::WeightedAverage());
	p_ov->setDefaultValue(fl::nan);
	p_ov->setPreviousValue(false);
	p_ov->addTerm(new fl::Constant("C1", 5));
	p_ov->addTerm(new fl::Constant("C2", 15));
	p_ov->addTerm(new fl::Constant("C3", 25));
	p_eng->addOutputVariable(p_ov);

	std::vector<fl::scalar> params;

	p_ov = new fl::OutputVariable();
	p_ov->setEnabled(true);
	p_ov->setName("Y2");
	p_ov->setRange(0, 100);
	p_ov->fuzzyOutput()->setAccumulation(fl::null);
	p_ov->setDefuzzifier(new fl::WeightedAverage());
	p_ov->setDefaultValue(fl::nan);
	p_ov->setPreviousValue(false);
	params.clear(); params.push_back(1); params.push_back(2); params.push_back(3);
	p_ov->addTerm(new fl::Linear("D1", params, p_eng));
	params.clear(); params.push_back(4); params.push_back(5); params.push_back(6);
	p_ov->addTerm(new fl::Linear("D2", params, p_eng));
	p_eng->addOutputVariable(p_ov);

	fl::RuleBlock* p_rules = new fl::RuleBlock();
	p_rules->setEnabled(true);
	p_rules->setConjunction(new fl::AlgebraicProduct());
	p_rules->setDisjunction(new fl::AlgebraicSum());
	p_rules->setActivation(fl::null);
	p_rules->addRule(fl::Rule::parse("if X1 is A1 and X2 is B1 then Y1 is C1 and Y2 is D1", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A1 and X2 is B2 then Y1 is C1 and Y2 is D2", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A2 and X2 is B1 then Y1 is C2 and Y2 is D1", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A2 and X2 is B2 then Y1 is C2 and Y2 is D2", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A3 and X2 is B1 then Y1 is C3 and Y2 is D1", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A3 and X2 is B2 then Y1 is C3 and Y2 is D2", p_eng));
	p_eng->addRuleBlock(p_rules);
}

void SetupMimoTsukamotoEngine(fl::Engine* p_eng)
{
	p_eng->setName("MIMO (Tsukamoto)");

	fl::InputVariable* p_iv = 0;

	p_iv = new fl::InputVariable();
	p_iv->setEnabled(true);
	p_iv->setName("X1");
	p_iv->setRange(0, 10);
	p_iv->addTerm(new fl::Gaussian("A1", 0, 1.5));
	p_iv->addTerm(new fl::Gaussian("A2", 5, 1.5));
	p_iv->addTerm(new fl::Gaussian("A3", 10, 1.5));
	p_eng->addInputVariable(p_iv);

	p_iv = new fl::InputVariable();
	p_iv->setEnabled(true);
	p_iv->setName("X2");
	p_iv->setRange(0, 10);
	p_iv->addTerm(new fl::Triangle("B1", 0, 0.5, 1));
	p_iv->addTerm(new fl::Triangle("B2", 0.5, 1, 1.5));
	p_eng->addInputVariable(p_iv);

	fl::OutputVariable* p_ov = 0;

	p_ov = new fl::OutputVariable();
	p_ov->setEnabled(true);
	p_ov->setName("Y1");
	p_ov->setRange(0, 30);
	p_ov->fuzzyOutput()->setAccumulation(fl::null);
	p_ov->setDefuzzifier(new fl::WeightedAverage());
	p_ov->setDefaultValue(fl::nan);
	p_ov->setPreviousValue(false);
	p_ov->addTerm(new fl::Concave("C1", 0.24, 0.25));
	p_ov->addTerm(new fl::Concave("C2", 0.5, 0.4));
	p_ov->addTerm(new fl::Concave("C3", 0.9, 1.0));
	p_eng->addOutputVariable(p_ov);

	p_ov = new fl::OutputVariable();
	p_ov->setEnabled(true);
	p_ov->setName("Y2");
	p_ov->setRange(0, 100);
	p_ov->fuzzyOutput()->setAccumulation(fl::null);
	p_ov->setDefuzzifier(new fl::WeightedAverage());
	p_ov->setDefaultValue(fl::nan);
	p_ov->setPreviousValue(false);
	p_ov->addTerm(new fl::SShape("D1", 0, 1));
	p_ov->addTerm(new fl::ZShape("D2", 0.5, 1.5));
	p_eng->addOutputVariable(p_ov);

	fl::RuleBlock* p_rules = new fl::RuleBlock();
	p_rules->setEnabled(true);
	p_rules->setConjunction(new fl::AlgebraicProduct());
	p_rules->setDisjunction(new fl::AlgebraicSum());
	p_rules->setActivation(fl::null);
	p_rules->addRule(fl::Rule::parse("if X1 is A1 and X2 is B1 then Y1 is C1 and Y2 is D1", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A1 and X2 is B2 then Y1 is C1 and Y2 is D2", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A2 and X2 is B1 then Y1 is C2 and Y2 is D1", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A2 and X2 is B2 then Y1 is C2 and Y2 is D2", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A3 and X2 is B1 then Y1 is C3 and Y2 is D1", p_eng));
	p_rules->addRule(fl::Rule::parse("if X1 is A3 and X2 is B2 then Y1 is C3 and Y2 is D2", p_eng));
	p_eng->addRuleBlock(p_rules);
}

bool CheckEqualValue(fl::scalar v1, fl::scalar v2, fl::scalar tol = 1e-9)
{
	if ((std::abs(v1-v2)/std::min(v1,v2)) > tol)
	{
		return false;
	}

	return true;
}

bool CheckEqualVariable(const fl::Variable& v1, const fl::Variable& v2)
{
	if (v1.getName() != v2.getName())
	{
		return false;
	}
	if (v1.numberOfTerms() != v2.numberOfTerms())
	{
		return false;
	}

	for (std::size_t j = 0,
					 nj = v1.numberOfTerms();
		 j < nj;
		 ++j)
	{
		const fl::Term* p_term1 = v1.getTerm(j);
		const fl::Term* p_term2 = v2.getTerm(j);

		if (p_term1->getName() != p_term2->getName())
		{
			return false;
		}
		if (p_term1->className() != p_term2->className())
		{
			return false;
		}
		if (p_term1->parameters() != p_term2->parameters())
		{
			return false;
		}
	}

	return true;
}

bool CheckEqualEvaluation(fl::Engine& eng1, fl::Engine& eng2)
{
	const std::size_t nv = 3;

	std::vector< std::vector<fl::scalar> > inValues;

	for (std::size_t i = 0,
					 ni = eng1.numberOfInputVariables();
		 i < ni;
		 ++i)
	{
		const fl::InputVariable* p_iv1 = eng1.getInputVariable(i);

		std::vector<fl::scalar> values(nv);
		values[0] = p_iv1->getMinimum();
		values[1] = (p_iv1->getMinimum()+p_iv1->getMaximum())/2.0;
		values[2] = p_iv1->getMaximum();

		inValues.push_back(values);
	}

	for (std::size_t i = 0; i < nv; ++i)
	{
		eng1.restart();
		eng2.restart();

		for (std::size_t j = 0,
						 nj = inValues.size();
			 j < nj;
			 ++j)
		{
			fl::InputVariable* p_iv1 = eng1.getInputVariable(j);
			fl::InputVariable* p_iv2 = eng2.getInputVariable(j);

			p_iv1->setValue(inValues[j][i]);
			p_iv2->setValue(inValues[j][i]);
		}

		eng1.process();
		eng2.process();

		for (std::size_t j = 0,
						 nj = eng1.numberOfOutputVariables();
			 j < nj;
			 ++j)
		{
			fl::OutputVariable* p_ov1 = eng1.getOutputVariable(j);
			fl::OutputVariable* p_ov2 = eng2.getOutputVariable(j);

			if (!CheckEqualValue(p_ov1->getValue(), p_ov2->getValue()))
			{
				return false;
			}
		}
	}

	return true;
}

bool CheckEqualEngine(const fl::Engine& eng1, const fl::Engine& eng2)
{
	// Check properties
	if (eng1.getName() != eng2.getName())
	{
		return false;
	}

	// Check inputs
	if (eng1.numberOfInputVariables() != eng2.numberOfInputVariables())
	{
		return false;
	}
	for (std::size_t i = 0,
					 ni = eng1.numberOfInputVariables();
		 i < ni;
		 ++i)
	{
		const fl::InputVariable* p_iv1 = eng1.getInputVariable(i);
		const fl::InputVariable* p_iv2 = eng2.getInputVariable(i);

		if (!CheckEqualVariable(*p_iv1, *p_iv2))
		{
			return false;
		}
	}

	// Check outputs
	if (eng1.numberOfOutputVariables() != eng2.numberOfOutputVariables())
	{
		return false;
	}
	for (std::size_t i = 0,
					 ni = eng1.numberOfOutputVariables();
		 i < ni;
		 ++i)
	{
		const fl::OutputVariable* p_ov1 = eng1.getOutputVariable(i);
		const fl::OutputVariable* p_ov2 = eng2.getOutputVariable(i);

		if (!CheckEqualVariable(*p_ov1, *p_ov2))
		{
			return false;
		}
	}

	// Check rules
	if (eng1.numberOfRuleBlocks() != eng2.numberOfRuleBlocks())
	{
		return false;
	}
	for (std::size_t i = 0,
					 ni = eng1.numberOfRuleBlocks();
		 i < ni;
		 ++i)
	{
		const fl::RuleBlock* p_rb1 = eng1.getRuleBlock(i);
		const fl::RuleBlock* p_rb2 = eng2.getRuleBlock(i);

		if (p_rb1->getName() != p_rb2->getName())
		{
			return false;
		}
		if (p_rb1->numberOfRules() != p_rb2->numberOfRules())
		{
			return false;
		}
		for (std::size_t j = 0,
						 nj = p_rb1->numberOfRules();
			 j < nj;
			 ++j)
		{
			const fl::Rule* p_r1 = p_rb1->getRule(i);
			const fl::Rule* p_r2 = p_rb2->getRule(i);

			if (p_r1->getText() != p_r2->getText())
			{
				return false;
			}
		}
	}

	return true;
}

bool CheckEqualEngine(const fl::anfis::Engine& eng1, const fl::anfis::Engine& eng2)
{
	if (!CheckEqualEngine(static_cast<const fl::Engine&>(eng1), static_cast<const fl::Engine&>(eng2)))
	{
		return false;
	}

	return true;
}

} // Namespace detail


void TestConstruction()
{
	// Default ctor
	{
		fl::anfis::Engine eng;

		if (eng.getName() != ""
			|| eng.numberOfInputVariables() != 0
			|| eng.numberOfOutputVariables() != 0
			|| eng.numberOfRuleBlocks() != 0)
		{
			throw std::runtime_error("Failed construction test: default construction");
		}
	}

	// 1-arg ctor
	{
		fl::anfis::Engine eng("anfis");

		if (eng.getName() != "anfis"
			|| eng.numberOfInputVariables() != 0
			|| eng.numberOfOutputVariables() != 0
			|| eng.numberOfRuleBlocks() != 0)
		{
			throw std::runtime_error("Failed construction test: 1-arg construction");
		}
	}

	// n-arg ctor
	{
		std::vector<fl::InputVariable*> inputs;
		fl::InputVariable* p_iv = new fl::InputVariable("X");
		p_iv->addTerm(new fl::Constant("A", 0));
		inputs.push_back(p_iv);

		std::vector<fl::OutputVariable*> outputs;
		fl::OutputVariable* p_ov = new fl::OutputVariable("Y");
		p_ov->setDefuzzifier(new fl::WeightedAverage());
		p_ov->addTerm(new fl::Constant("B", 0));
		outputs.push_back(p_ov);

		std::vector<fl::RuleBlock*> ruleblocks;
		fl::RuleBlock* p_rb = new fl::RuleBlock();
		p_rb->addRule(new fl::Rule("if X is A then Y is B"));
		ruleblocks.push_back(p_rb);

		fl::anfis::Engine eng(inputs.begin(), inputs.end(),
							  outputs.begin(), outputs.end(),
							  ruleblocks.begin(), ruleblocks.end(),
							  "anfis");

		if (eng.getName() != "anfis"
			|| eng.numberOfInputVariables() != 1
			|| eng.numberOfOutputVariables() != 1
			|| eng.numberOfRuleBlocks() != 1)
		{
			throw std::runtime_error("Failed construction test: n-arg construction");
		}
	}
}

/// Test copy construction & similar functionalities
void TestCopy()
{
	// Copy ctor (from ANFIS engine)
	{
		fl::anfis::Engine eng1;

		detail::SetupMimoSugenoEngine(&eng1);
		eng1.build();

		fl::anfis::Engine eng2(eng1);

		if (!detail::CheckEqualEngine(eng1, eng2))
		{
			throw std::runtime_error("Failed copy test: copy-construction");
		}
	}

	// Alternativ copy ctor (from base class engine)
	{
		fl::Engine eng1;

		detail::SetupMimoSugenoEngine(&eng1);

		fl::anfis::Engine eng2(eng1);

		if (!detail::CheckEqualEngine(eng1, eng2))
		{
			throw std::runtime_error("Failed copy test: alternative copy-construction");
		}
	}

	// Assignment operator
	{
		fl::anfis::Engine eng1;

		detail::SetupMimoSugenoEngine(&eng1);
		eng1.build();

		fl::anfis::Engine eng2;

		eng2 = eng1;

		if (!detail::CheckEqualEngine(eng1, eng2))
		{
			throw std::runtime_error("Failed copy test: copy-assignment");
		}
	}

	// Cloning
	{
		fl::anfis::Engine eng1;

		detail::SetupMimoSugenoEngine(&eng1);
		eng1.build();

		FL_unique_ptr<fl::anfis::Engine> p_eng2(eng1.clone());

		if (!detail::CheckEqualEngine(eng1, *p_eng2))
		{
			throw std::runtime_error("Failed copy test: cloning");
		}
	}
}

/// Test functional FIS behavior
void TestFunctional()
{
	// An ANFIS should be just like a traditional FIS

	{
		fl::Engine eng;
		detail::SetupMimoSugenoEngine(&eng);

		fl::anfis::Engine anfis;
		detail::SetupMimoSugenoEngine(&anfis);
		anfis.build();

		if (!detail::CheckEqualEvaluation(eng, anfis))
		{
			throw std::runtime_error("Failed functional test: Takagi-Sugeno evaluation");
		}
	}

	{
		fl::Engine eng;
		detail::SetupMimoTsukamotoEngine(&eng);

		fl::anfis::Engine anfis;
		detail::SetupMimoTsukamotoEngine(&anfis);
		anfis.build();

		if (!detail::CheckEqualEvaluation(eng, anfis))
		{
			throw std::runtime_error("Failed functional test: Tsukamoto evaluation");
		}
	}
}

} // Namespace <unnamed>


int main()
{
	try
	{
		std::cout << "- Testing construction... ";
		TestConstruction();
		std::cout << "OK";
	}
	catch (const std::exception& e)
	{
		std::cout << "KO => " << e.what();
	}
	catch (...)
	{
		std::cout << "KO => unexpected error";
	}
	std::cout << std::endl;

	try
	{
		std::cout << "- Testing copy construction & similar functionalities... ";
		TestCopy();
		std::cout << "OK";
	}
	catch (const std::exception& e)
	{
		std::cout << "KO => " << e.what();
	}
	catch (...)
	{
		std::cout << "KO => unexpected error";
	}
	std::cout << std::endl;

	try
	{
		std::cout << "- Testing functional behavior... ";
		TestFunctional();
		std::cout << "OK";
	}
	catch (const std::exception& e)
	{
		std::cout << "KO => " << e.what();
	}
	catch (...)
	{
		std::cout << "KO => unexpected error";
	}
	std::cout << std::endl;
}
