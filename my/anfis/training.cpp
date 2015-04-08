#include <cmath>
#include <cstddef>
#include <deque>
#include <fl/anfis/engine.h>
#include <fl/anfis/training.h>
#include <fl/dataset.h>
#include <fl/detail/math.h>
#include <fl/detail/rls.h>
#include <fl/detail/terms.h>
#include <fl/detail/traits.h>
#include <fl/fuzzylite.h>
#include <fl/Operation.h>
#include <fl/term/Term.h>
#include <fl/variable/OutputVariable.h>
#include <map>
#include <vector>


namespace fl { namespace anfis {

Jang1993HybridLearningAlgorithm::Jang1993HybridLearningAlgorithm(Engine* p_anfis,
																 fl::scalar ss,
																 fl::scalar ssDecrRate,
																 fl::scalar ssIncrRate,
																 fl::scalar ff)
: p_anfis_(p_anfis),
  stepSizeInit_(ss),
  stepSizeDecrRate_(ssDecrRate),
  stepSizeIncrRate_(ssIncrRate),
  stepSizeErrWindowLen_(5),
  stepSizeIncrCounter_(0),
  stepSizeDecrCounter_(0),
  online_(false),
  //useBias_(true),
  rls_(0,0,0,ff)
{
	this->init();
}

void Jang1993HybridLearningAlgorithm::setEngine(Engine* p_anfis)
{
	p_anfis_ = p_anfis;
}

Engine* Jang1993HybridLearningAlgorithm::getEngine() const
{
	return p_anfis_;
}

void Jang1993HybridLearningAlgorithm::setInitialStepSize(fl::scalar value)
{
	stepSizeInit_ = value;
}

fl::scalar Jang1993HybridLearningAlgorithm::getInitialStepSize() const
{
	return stepSizeInit_;
}

void Jang1993HybridLearningAlgorithm::setStepSizeDecreaseRate(fl::scalar value)
{
	stepSizeDecrRate_ = value;
}

fl::scalar Jang1993HybridLearningAlgorithm::getStepSizeDecreaseRate() const
{
	return stepSizeDecrRate_;
}

void Jang1993HybridLearningAlgorithm::setStepSizeIncreaseRate(fl::scalar value)
{
	stepSizeIncrRate_ = value;
}

fl::scalar Jang1993HybridLearningAlgorithm::getStepSizeIncreaseRate() const
{
	return stepSizeIncrRate_;
}

void Jang1993HybridLearningAlgorithm::setForgettingFactor(fl::scalar value)
{
	rls_.setForgettingFactor(value);
}

fl::scalar Jang1993HybridLearningAlgorithm::getForgettingFactor() const
{
	return rls_.getForgettingFactor();
}

void Jang1993HybridLearningAlgorithm::setIsOnline(bool value)
{
	online_ = value;
}

bool Jang1993HybridLearningAlgorithm::isOnline() const
{
	return online_;
}

fl::scalar Jang1993HybridLearningAlgorithm::train(const fl::DataSet<fl::scalar>& data,
												  std::size_t maxEpochs,
												  fl::scalar errorGoal)
{
	this->reset();

	fl::scalar rmse = 0;
	for (std::size_t epoch = 0; epoch < maxEpochs; ++epoch)
	{
		FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch);

		rmse = this->trainSingleEpoch(data);

		FL_DEBUG_TRACE("TRAINING - EPOCH #" << epoch << " -> RMSE: " << rmse);

		if (fl::detail::FloatTraits<fl::scalar>::EssentiallyLessEqual(rmse, errorGoal))
		{
			break;
		}
	}
	return rmse;
}

fl::scalar Jang1993HybridLearningAlgorithm::trainSingleEpoch(const fl::DataSet<fl::scalar>& data)
{
	this->check();

	p_anfis_->setIsLearning(true);

	// Update parameters of input terms
	if (dEdPs_.size() > 0)
	{
		std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
		const std::size_t ni = fuzzyLayer.size();

		fl::scalar errNorm = 0;
		for (std::size_t i = 0; i < ni; ++i)
		{
			FuzzificationNode* p_node = fuzzyLayer[i];

//std::cerr << "PHASE #-1 - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX

			for (std::size_t p = 0,
							 np = dEdPs_.at(p_node).size();
				 p < np;
				 ++p)
			{
				errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
			}
		}
		errNorm = std::sqrt(errNorm);
//std::cerr << "PHASE #-1 - Error Norm: " << errNorm << std::endl;///XXX
//std::cerr << "PHASE #-1 - STEP-SIZE: " << stepSize_ << std::endl;///XXX
		if (errNorm > 0)
		{
			for (std::size_t i = 0; i < ni; ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];
				std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());

//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
				for (std::size_t p = 0,
								 np = dEdPs_.at(p_node).size();
					 p < np;
					 ++p)
				{
					params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
				}
//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
				detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
			}
		}
	}
	// Update step-size
//std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
//std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
	{
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeDecrCounter_ >= maxCounter)
		{
//std::cerr << "STEP-SIZE decrease checking..." << std::endl;//XXX
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				if ((i % 2) != 0)
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] <= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
				}
				else
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
				}
			}
			if (update)
			{
//std::cerr << "STEP-SIZE (decreasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeDecrRate_) << std::endl;//XXX
				stepSize_ *= stepSizeDecrRate_;
				stepSizeDecrCounter_ = 1;
			}
			else
			{
				++stepSizeDecrCounter_;
			}
		}
		else
		{
			++stepSizeDecrCounter_;
		}
	}
	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeIncrRate_, 1))
	{
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeIncrCounter_ >= maxCounter)
		{
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				////update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
				//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
				if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
				{
					update = false;
				}
			}
			if (update)
			{
//std::cerr << "STEP-SIZE (increasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeIncrRate_) << std::endl;//XXX
				stepSize_ *= stepSizeIncrRate_;
				stepSizeIncrCounter_ = 1;
			}
			else
			{
				++stepSizeIncrCounter_;
			}
		}
		else
		{
			++stepSizeIncrCounter_;
		}
	}

	rls_.reset();
	dEdPs_.clear();
	//if (rlsPhi_.size() > 0)
	//{
	//	// Restore the RLS regressor vector of the previous epoch
	//	rls_.setRegressor(rlsPhi_);
	//}
	//stepSize_ = stepSizeInit_;
	//stepSizeErrWindow_.clear();

	fl::scalar rmse = 0;

	if (online_)
	{
		rmse = this->trainSingleEpochOnline(data);
	}
	else
	{
		rmse = this->trainSingleEpochOffline(data);
	}

	if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
	{
		stepSizeErrWindow_.pop_back();
		//stepSizeErrWindow_.pop_front();
	}
	stepSizeErrWindow_.push_front(rmse);
	//stepSizeErrWindow_.push_back(rmse);

//[XXX]: Moved at the beginning of this method
//	// Update parameters of input terms
//	if (dEdPs_.size() > 0)
//	{
//		std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
//		const std::size_t ni = fuzzyLayer.size();
//
//		fl::scalar errNorm = 0;
//		for (std::size_t i = 0; i < ni; ++i)
//		{
//			FuzzificationNode* p_node = fuzzyLayer[i];
//
////std::cerr << "PHASE #2 - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX
//			for (std::size_t p = 0,
//							 np = dEdPs_.at(p_node).size();
//				 p < np;
//				 ++p)
//			{
//				errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
//			}
//		}
//		errNorm = std::sqrt(errNorm);
////std::cerr << "PHASE #2 - Error Norm: " << errNorm << std::endl;///XXX
////std::cerr << "PHASE #2 - Step Size: " << stepSize_ << std::endl;///XXX
//		if (errNorm > 0)
//		{
//			for (std::size_t i = 0; i < ni; ++i)
//			{
//				FuzzificationNode* p_node = fuzzyLayer[i];
//				std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());
//
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
////std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
//				for (std::size_t p = 0,
//								 np = dEdPs_.at(p_node).size();
//					 p < np;
//					 ++p)
//				{
//					params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
//				}
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//				detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
//			}
//		}
//	}
//
//	// Update step-size
////std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
////std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
//	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
//	{
//		if (stepSizeDecrCounter_ == (stepSizeErrWindowLen_-1))
//		{
//			bool update = true;
//			for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
//			{
//				if (i % 2)
//				{
//					//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
//					update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
//				}
//				else
//				{
//					//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
//					update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
//				}
//			}
//			if (update)
//			{
////std::cerr << "STEP-SIZE (decreasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeDecrRate_) << std::endl;//XXX
//				stepSize_ *= stepSizeDecrRate_;
//				stepSizeDecrCounter_ = 1;
//			}
//			else
//			{
//				++stepSizeDecrCounter_;
//			}
//		}
//		else
//		{
//			++stepSizeDecrCounter_;
//		}
//	}
//	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeIncrRate_, 1))
//	{
//		if (stepSizeIncrCounter_ == (stepSizeErrWindowLen_-1))
//		{
//			bool update = true;
//			for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
//			{
//				//update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
//				update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
//			}
//			if (update)
//			{
////std::cerr << "STEP-SIZE (increasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeIncrRate_) << std::endl;//XXX
//				stepSize_ *= stepSizeIncrRate_;
//				stepSizeIncrCounter_ = 1;
//			}
//			else
//			{
//				++stepSizeIncrCounter_;
//			}
//		}
//		else
//		{
//			++stepSizeIncrCounter_;
//		}
//	}
//[/XXX]: Moved at the beginning of this method

	p_anfis_->setIsLearning(false);

	return rmse;
}


fl::scalar Jang1993HybridLearningAlgorithm::trainSingleEpochOffline(const fl::DataSet<fl::scalar>& data)
{
/*
	this->check();

	p_anfis_->setIsLearning(true);

	// Update parameters of input terms
	if (dEdPs_.size() > 0)
	{
		std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
		const std::size_t ni = fuzzyLayer.size();

		fl::scalar errNorm = 0;
		for (std::size_t i = 0; i < ni; ++i)
		{
			FuzzificationNode* p_node = fuzzyLayer[i];

//std::cerr << "PHASE #-1 - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX

			for (std::size_t p = 0,
							 np = dEdPs_.at(p_node).size();
				 p < np;
				 ++p)
			{
				errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
			}
		}
		errNorm = std::sqrt(errNorm);
//std::cerr << "PHASE #-1 - Error Norm: " << errNorm << std::endl;///XXX
//std::cerr << "PHASE #-1 - STEP-SIZE: " << stepSize_ << std::endl;///XXX
		if (errNorm > 0)
		{
			for (std::size_t i = 0; i < ni; ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];
				std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());

//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
				for (std::size_t p = 0,
								 np = dEdPs_.at(p_node).size();
					 p < np;
					 ++p)
				{
					params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
				}
//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
				detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
			}
		}
	}
	// Update step-size
//std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
//std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
	{
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeDecrCounter_ >= maxCounter)
		{
//std::cerr << "STEP-SIZE decrease checking..." << std::endl;//XXX
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				if ((i % 2) != 0)
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] <= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
				}
				else
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
				}
			}
			if (update)
			{
//std::cerr << "STEP-SIZE (decreasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeDecrRate_) << std::endl;//XXX
				stepSize_ *= stepSizeDecrRate_;
				stepSizeDecrCounter_ = 1;
			}
			else
			{
				++stepSizeDecrCounter_;
			}
		}
		else
		{
			++stepSizeDecrCounter_;
		}
	}
	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeIncrRate_, 1))
	{
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeIncrCounter_ >= maxCounter)
		{
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				////update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
				//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
				if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
				{
					update = false;
				}
			}
			if (update)
			{
//std::cerr << "STEP-SIZE (increasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeIncrRate_) << std::endl;//XXX
				stepSize_ *= stepSizeIncrRate_;
				stepSizeIncrCounter_ = 1;
			}
			else
			{
				++stepSizeIncrCounter_;
			}
		}
		else
		{
			++stepSizeIncrCounter_;
		}
	}

	rls_.reset();
	dEdPs_.clear();
	//if (rlsPhi_.size() > 0)
	//{
	//	// Restore the RLS regressor vector of the previous epoch
	//	rls_.setRegressor(rlsPhi_);
	//}
	//stepSize_ = stepSizeInit_;
	//stepSizeErrWindow_.clear();
*/

	fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch

	// Forwards inputs from input layer to antecedent layer, and estimate parameters with RLS
	//std::vector< std::vector<fl::scalar> > antecedentValues;
	for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
		 entryIt != entryEndIt;
		 ++entryIt)
	{
		const fl::DataSetEntry<fl::scalar>& entry = *entryIt;

		const std::size_t nout = entry.numOfOutputs();

		if (nout != p_anfis_->numberOfOutputVariables())
		{
			FL_THROW2(std::invalid_argument, "Incorrect output dimension");
		}

		const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

		// Compute current rule firing strengths
		const std::vector<fl::scalar> ruleFiringStrengths = p_anfis_->evalTo(entry.inputBegin(), entry.inputEnd(), fl::anfis::Engine::AntecedentLayer);

//std::cerr << "PHASE #0 - Traning data #: " << std::distance(data.entryBegin(), entryIt) << std::endl;//XXX
//std::cerr << "PHASE #0 - Entry input: "; fl::detail::VectorOutput(std::cerr, std::vector<fl::scalar>(entry.inputBegin(), entry.inputEnd())); std::cerr << std::endl;//XXX
////std::cerr << "PHASE #0 - Rule firing strength: "; fl::detail::VectorOutput(std::cerr, ruleFiringStrengths); std::cerr << std::endl;//XXX
//{[XXX]
//	std::vector< std::vector<fl::scalar> > mfParams;
//	std::cerr << "PHASE #0 - Layer 0: [";
//	std::vector<InputNode*> nodes0 = p_anfis_->getInputLayer();
//	for (std::size_t i = 0; i < nodes0.size(); ++i)
//	{
//		std::cerr << nodes0[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 1: [";
//	std::vector<FuzzificationNode*> nodes1 = p_anfis_->getFuzzificationLayer();
//	for (std::size_t i = 0; i < nodes1.size(); ++i)
//	{
//		std::cerr << nodes1[i]->getValue() << " ";
//		mfParams.push_back(detail::GetTermParameters(nodes1[i]->getTerm()));
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 2: ";
//	std::vector<InputHedgeNode*> nodes2 = p_anfis_->getInputHedgeLayer();
//	for (std::size_t i = 0; i < nodes2.size(); ++i)
//	{
//		std::cerr << nodes2[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//	std::cerr << "PHASE #0 - Layer 3: ";
//	std::vector<AntecedentNode*> nodes3 = p_anfis_->getAntecedentLayer();
//	for (std::size_t i = 0; i < nodes3.size(); ++i)
//	{
//		std::cerr << nodes3[i]->getValue() << " ";
//	}
//	std::cerr << "]" << std::endl;
//
//	for (std::size_t i = 0; i < mfParams.size(); ++i)
//	{
//		std::cerr << "MF #" << i << " Parameters: "; fl::detail::VectorOutput(std::cerr, mfParams[i]); std::cerr << std::endl;
//	}
//}[/XXX]
		// Compute normalization factor
		const fl::scalar totRuleFiringStrength = fl::detail::Sum<fl::scalar>(ruleFiringStrengths.begin(), ruleFiringStrengths.end());
//{//[XXX]
//std::cerr << "PHASE #0 - Total Rule firing strength: " << totRuleFiringStrength << std::endl;
//std::cerr << "PHASE #0 - Normalized Rule firing strength: ";
//for (std::size_t i = 0; i < ruleFiringStrengths.size(); ++i)
//{
//	std::cerr << ruleFiringStrengths[i]/totRuleFiringStrength << " ";
//}
//std::cerr << std::endl;
//}//[XXX]
		// Compute input to RLS algorithm
		std::vector<fl::scalar> rlsInputs(rls_.getInputDimension());
		{
			std::size_t k = 0;
			std::size_t r = 0;
			for (std::size_t v = 0,
							 nv = p_anfis_->numberOfOutputVariables();
				 v < nv;
				 ++v)
			{
				fl::OutputVariable* p_var = p_anfis_->getOutputVariable(v);

				FL_DEBUG_ASSERT( p_var );

				for (std::size_t t = 0,
								 nt = p_var->numberOfTerms();
					 t < nt;
					 ++t)
				{
					fl::Term* p_term = p_var->getTerm(t);

					FL_DEBUG_ASSERT( p_term );

					const std::size_t numParams = detail::GetTermParameters(p_term).size();
					for (std::size_t p = 1; p < numParams; ++p)
					{
						rlsInputs[k] = ruleFiringStrengths[r]*entry.getField(p-1)/totRuleFiringStrength;
						++k;
					}
					rlsInputs[k] = ruleFiringStrengths[r]/totRuleFiringStrength;
					++k;
					++r;
				}
			}
		}
//std::cerr << "PHASE #0 - Num inputs: " << rls_.getInputDimension() << " - Num Outputs: " << rls_.getOutputDimension() << " - Order: " << rls_.getModelOrder() << std::endl;//XXX
//std::cerr << "PHASE #0 - RLS Input: "; fl::detail::VectorOutput(std::cerr, rlsInputs); std::cerr << std::endl;///XXX
		// Estimate parameters
		std::vector<fl::scalar> actualOut;
		actualOut = rls_.estimate(rlsInputs.begin(), rlsInputs.end(), targetOut.begin(), targetOut.end());
//std::cerr << "PHASE #0 - Target: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - Actual: ";fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << std::endl;///XXX
	}

	// Put estimated RLS parameters in the ANFIS model and save RLS regressor vector
	{
		const std::vector< std::vector<fl::scalar> > rlsParamMatrix = rls_.getEstimatedParameters();
//std::cerr << "PHASE #0 - Estimated RLS params: "; fl::detail::MatrixOutput(std::cerr, rlsParamMatrix); std::cerr << std::endl;//XXX

		std::size_t k = 0;
		//std::size_t r = 0;
		for (std::size_t v = 0,
						 nv = p_anfis_->numberOfOutputVariables();
			 v < nv;
			 ++v)
		{
			fl::OutputVariable* p_var = p_anfis_->getOutputVariable(v);

			FL_DEBUG_ASSERT( p_var );

			for (std::size_t t = 0,
							 nt = p_var->numberOfTerms();
				 t < nt;
				 ++t)
			{
				fl::Term* p_term = p_var->getTerm(t);

				FL_DEBUG_ASSERT( p_term );

				const std::size_t numParams = detail::GetTermParameters(p_term).size();
				std::vector<fl::scalar> params(numParams);
				for (std::size_t p = 0; p < numParams; ++p)
				{
					params[p] = rlsParamMatrix[k][v];
					++k;
				}
				detail::SetTermParameters(p_term, params.begin(), params.end());
//std::cerr << "PHASE #0 - Estimated RLS params - Output #" << v << " - Term #" << t << " - Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;//XXX
				//++r;
			}
		}

		//rlsPhi_ = rls_.getRegressor();
	}

	for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
		 entryIt != entryEndIt;
		 ++entryIt)
	{
		const fl::DataSetEntry<fl::scalar>& entry = *entryIt;
//std::cerr << "PHASE #1 - Traning data #: " << std::distance(data.entryBegin(), entryIt) << std::endl;//XXX

		const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

		// Compute ANFIS output
		const std::vector<fl::scalar> actualOut = p_anfis_->eval(entry.inputBegin(), entry.inputEnd());

		// Update bias in case of zero rule firing strength
		if (p_anfis_->hasBias())
		{
			bool skip = false;

			for (std::size_t i = 0,
							 ni = actualOut.size();
				 i < ni;
				 ++i)
			{
				if (fl::Operation::isNaN(actualOut[i]))
				{
					OutputNode* p_outNode = p_anfis_->getOutputLayer().at(i);

					FL_DEBUG_ASSERT( p_outNode );

					//bias_[i] += stepSize_*(targetOut[i]-bias_[i]);
					fl::scalar bias = p_outNode->getBias();
					bias += stepSize_*(targetOut[i]-bias);
					p_outNode->setBias(bias);
					skip = true;
				}
			}
			//p_anfis_->setBias(bias_);

			if (skip)
			{
				// Skip this data point
//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, bias_); std::cerr << std::endl; //XXX
				continue;
			}
		}

//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, bias_); std::cerr << std::endl; //XXX

		// Update error
		fl::scalar se = 0;
		for (std::size_t i = 0,
						 ni = targetOut.size();
			 i < ni;
			 ++i)
		{
			const fl::scalar out = fl::Operation::isNaN(actualOut[i]) ? 0.0 : actualOut[i];

			se += fl::detail::Sqr(targetOut[i]-out);
		}
		rmse += se;
//std::cerr << "PHASE #1 - Current error: " <<  se << " - Total error: " << rmse << std::endl;//XXX

		// Backward errors
		std::map<const Node*,fl::scalar> dEdOs;
		// Computes error derivatives at output layer
		{
			const std::vector<OutputNode*> outLayer = p_anfis_->getOutputLayer();
			for (std::size_t i = 0,
							 ni = targetOut.size();
				 i < ni;
				 ++i)
			{
				const Node* p_node = outLayer[i];

				dEdOs[p_node] = -2.0*(targetOut[i]-actualOut[i]);
//std::cerr << "PHASE #1 - Layer: " << Engine::OutputLayer << ", Node: " << i << ", dEdO: " << dEdOs[p_node] << std::endl;//XXX
			}
		}
		// Propagates errors back to the fuzzification layer
		for (Engine::LayerCategory layerCat = p_anfis_->getPreviousLayerCategory(Engine::OutputLayer);
			 layerCat != Engine::InputLayer;
			 layerCat = p_anfis_->getPreviousLayerCategory(layerCat))
		{
			std::vector<Node*> layer = p_anfis_->getLayer(layerCat);

			for (std::size_t i = 0,
							 ni = layer.size();
				 i < ni;
				 ++i)
			{
				Node* p_fromNode = layer[i];

				fl::scalar dEdO = 0;
				std::vector<Node*> outConns = p_fromNode->outputConnections();
				for (std::size_t j = 0,
								 nj = outConns.size();
					 j < nj;
					 ++j)
				{
					Node* p_toNode = outConns[j];

					const std::vector<fl::scalar> dOdOs = p_toNode->evalDerivativeWrtInputs();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - dOdOs: "; fl::detail::VectorOutput(std::cerr, dOdOs); std::cerr << std::endl;//XXX

					// Find the index k in the input connection of p_fromNode related to the input node p_toNode
					const std::vector<Node*> inConns = p_toNode->inputConnections();
					const std::size_t nk = inConns.size();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - #In Conns: " << nk << " - double check #In Conns: " << p_anfis_->inputConnections(p_toNode).size() << std::endl;//XXX
					std::size_t k = 0;
					while (k < nk && inConns[k] != p_fromNode)
					{
						++k;
					}
					if (k == nk)
					{
						FL_THROW2(std::runtime_error, "Found inconsistencies in input and output connections");
					}
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", Node: " << i << " - Found k: " << k << std::endl;//XXX

					dEdO += dEdOs[p_toNode]*dOdOs[k];
				}
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", Node: " << i << "..." << std::endl;//XXX
				dEdOs[p_fromNode] = dEdO;
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << ", dEdO: " << dEdOs[p_fromNode] << std::endl;//XXX
			}
		}

//std::cerr << "PHASE #1 - Updating parameters" << std::endl;
		// Update error derivatives wrt parameters $\frac{\partial E}{\partial P_{ij}}$
		std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
		for (std::size_t i = 0,
						 ni = fuzzyLayer.size();
			 i < ni;
			 ++i)
		{
			FuzzificationNode* p_node = fuzzyLayer[i];

			const std::vector<fl::scalar> dOdPs = p_node->evalDerivativeWrtParams();
//std::cerr << "PHASE #1 - Layer: " << Engine::FuzzificationLayer << ", Node: " << i << " (" << p_node << "), dOdPs: "; fl::detail::VectorOutput(std::cerr, dOdPs); std::cerr << std::endl;//XXX
			const std::size_t np = dOdPs.size();

			if (dEdPs_.count(p_node) == 0)
			{
				dEdPs_[p_node].resize(np, 0);
			}

			for (std::size_t p = 0; p < np; ++p)
			{
				dEdPs_[p_node][p] += dEdOs[p_node]*dOdPs[p];
//std::cerr << "PHASE #1 - Layer: fuzzification, Node: " << i << " (" << p_node << "), dEdP_" << p << ": " << dEdPs_.at(p_node)[p] << std::endl;//XXX
			}
		}
	}

	rmse = std::sqrt(rmse/data.size());

/*
	if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
	{
		stepSizeErrWindow_.pop_back();
		//stepSizeErrWindow_.pop_front();
	}
	stepSizeErrWindow_.push_front(rmse);
	//stepSizeErrWindow_.push_back(rmse);

//[XXX]: Moved at the beginning of this method
//	// Update parameters of input terms
//	if (dEdPs_.size() > 0)
//	{
//		std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
//		const std::size_t ni = fuzzyLayer.size();
//
//		fl::scalar errNorm = 0;
//		for (std::size_t i = 0; i < ni; ++i)
//		{
//			FuzzificationNode* p_node = fuzzyLayer[i];
//
////std::cerr << "PHASE #2 - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX
//			for (std::size_t p = 0,
//							 np = dEdPs_.at(p_node).size();
//				 p < np;
//				 ++p)
//			{
//				errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
//			}
//		}
//		errNorm = std::sqrt(errNorm);
////std::cerr << "PHASE #2 - Error Norm: " << errNorm << std::endl;///XXX
////std::cerr << "PHASE #2 - Step Size: " << stepSize_ << std::endl;///XXX
//		if (errNorm > 0)
//		{
//			for (std::size_t i = 0; i < ni; ++i)
//			{
//				FuzzificationNode* p_node = fuzzyLayer[i];
//				std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());
//
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
////std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
//				for (std::size_t p = 0,
//								 np = dEdPs_.at(p_node).size();
//					 p < np;
//					 ++p)
//				{
//					params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
//				}
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//				detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
//			}
//		}
//	}
//
//	// Update step-size
////std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
////std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
//	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
//	{
//		if (stepSizeDecrCounter_ == (stepSizeErrWindowLen_-1))
//		{
//			bool update = true;
//			for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
//			{
//				if (i % 2)
//				{
//					//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
//					update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
//				}
//				else
//				{
//					//update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
//					update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
//				}
//			}
//			if (update)
//			{
////std::cerr << "STEP-SIZE (decreasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeDecrRate_) << std::endl;//XXX
//				stepSize_ *= stepSizeDecrRate_;
//				stepSizeDecrCounter_ = 1;
//			}
//			else
//			{
//				++stepSizeDecrCounter_;
//			}
//		}
//		else
//		{
//			++stepSizeDecrCounter_;
//		}
//	}
//	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeIncrRate_, 1))
//	{
//		if (stepSizeIncrCounter_ == (stepSizeErrWindowLen_-1))
//		{
//			bool update = true;
//			for (std::size_t i = 0; i < (stepSizeErrWindowLen_-1); ++i)
//			{
//				//update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
//				update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
//			}
//			if (update)
//			{
////std::cerr << "STEP-SIZE (increasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeIncrRate_) << std::endl;//XXX
//				stepSize_ *= stepSizeIncrRate_;
//				stepSizeIncrCounter_ = 1;
//			}
//			else
//			{
//				++stepSizeIncrCounter_;
//			}
//		}
//		else
//		{
//			++stepSizeIncrCounter_;
//		}
//	}
//[/XXX]: Moved at the beginning of this method

	p_anfis_->setIsLearning(false);
*/

	return rmse;
}

fl::scalar Jang1993HybridLearningAlgorithm::trainSingleEpochOnline(const fl::DataSet<fl::scalar>& data)
{
/*
	this->check();

	p_anfis_->setIsLearning(true);

	// Update parameters of input terms
	if (dEdPs_.size() > 0)
	{
		std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
		const std::size_t ni = fuzzyLayer.size();

		fl::scalar errNorm = 0;
		for (std::size_t i = 0; i < ni; ++i)
		{
			FuzzificationNode* p_node = fuzzyLayer[i];

//std::cerr << "PHASE #-1 - Computing Error Norm for node #" << i << " (" << p_node << ")... "<< std::endl;///XXX

			for (std::size_t p = 0,
							 np = dEdPs_.at(p_node).size();
				 p < np;
				 ++p)
			{
				errNorm += fl::detail::Sqr(dEdPs_.at(p_node).at(p));
			}
		}
		errNorm = std::sqrt(errNorm);
//std::cerr << "PHASE #-1 - Error Norm: " << errNorm << std::endl;///XXX
//std::cerr << "PHASE #-1 - STEP-SIZE: " << stepSize_ << std::endl;///XXX
		if (errNorm > 0)
		{
			for (std::size_t i = 0; i < ni; ++i)
			{
				FuzzificationNode* p_node = fuzzyLayer[i];
				std::vector<fl::scalar> params = detail::GetTermParameters(p_node->getTerm());

//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - Old Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
//std::cerr << "PHASE #2 - Node #" << i << ": " << p_node << " - dEdPs: "; fl::detail::VectorOutput(std::cerr, dEdPs_.at(p_node)); std::cerr << std::endl;///XXX
				for (std::size_t p = 0,
								 np = dEdPs_.at(p_node).size();
					 p < np;
					 ++p)
				{
					params[p] -= stepSize_*dEdPs_.at(p_node).at(p)/errNorm;
				}
//std::cerr << "PHASE #-1 - Node #" << i << ": " << p_node << " - New Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;///XXX
				detail::SetTermParameters(p_node->getTerm(), params.begin(), params.end());
			}
		}
	}
	// Update step-size
//std::cerr << "STEP-SIZE error window: "; fl::detail::VectorOutput(std::cerr, stepSizeErrWindow_); std::cerr << std::endl;//XXX
//std::cerr << "STEP-SIZE decr-counter: " << stepSizeDecrCounter_ << ", incr-counter: " << stepSizeIncrCounter_ << std::endl;//XXX
	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeDecrRate_, 1))
	{
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeDecrCounter_ >= maxCounter)
		{
//std::cerr << "STEP-SIZE decrease checking..." << std::endl;//XXX
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				if ((i % 2) != 0)
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] > stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] > stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] <= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
				}
				else
				{
					////update &= (stepSizeErrWindow_[stepSizeDecrCounter_-i] < stepSizeErrWindow_[stepSizeDecrCounter_-i-1]);
					//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
					if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
					{
						update = false;
					}
				}
			}
			if (update)
			{
//std::cerr << "STEP-SIZE (decreasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeDecrRate_) << std::endl;//XXX
				stepSize_ *= stepSizeDecrRate_;
				stepSizeDecrCounter_ = 1;
			}
			else
			{
				++stepSizeDecrCounter_;
			}
		}
		else
		{
			++stepSizeDecrCounter_;
		}
	}
	if (!fl::detail::FloatTraits<fl::scalar>::EssentiallyEqual(stepSizeIncrRate_, 1))
	{
		const std::size_t maxCounter = stepSizeErrWindowLen_-1;

		if (stepSizeErrWindow_.size() >= stepSizeErrWindowLen_
			&& stepSizeIncrCounter_ >= maxCounter)
		{
			bool update = true;
			for (std::size_t i = 0; i < maxCounter && update; ++i)
			{
				////update &= (stepSizeErrWindow_[stepSizeIncrCounter_-i] < stepSizeErrWindow_[stepSizeIncrCounter_-i-1]);
				//update &= (stepSizeErrWindow_[i] < stepSizeErrWindow_[i+1]);
				if (stepSizeErrWindow_[i] >= stepSizeErrWindow_[i+1])
				{
					update = false;
				}
			}
			if (update)
			{
//std::cerr << "STEP-SIZE (increasing) - old: " << stepSize_ << ", new: " << (stepSize_*stepSizeIncrRate_) << std::endl;//XXX
				stepSize_ *= stepSizeIncrRate_;
				stepSizeIncrCounter_ = 1;
			}
			else
			{
				++stepSizeIncrCounter_;
			}
		}
		else
		{
			++stepSizeIncrCounter_;
		}
	}

	rls_.reset();
	dEdPs_.clear();
	//if (rlsPhi_.size() > 0)
	//{
	//	// Restore the RLS regressor vector of the previous epoch
	//	rls_.setRegressor(rlsPhi_);
	//}
	//stepSize_ = stepSizeInit_;
	//stepSizeErrWindow_.clear();
*/

	fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE) for this epoch

	// Forwards inputs from input layer to antecedent layer, and estimate parameters with RLS
	//std::vector< std::vector<fl::scalar> > antecedentValues;
	for (typename fl::DataSet<fl::scalar>::ConstEntryIterator entryIt = data.entryBegin(),
															  entryEndIt = data.entryEnd();
		 entryIt != entryEndIt;
		 ++entryIt)
	{
		const fl::DataSetEntry<fl::scalar>& entry = *entryIt;

		const std::size_t nout = entry.numOfOutputs();

		if (nout != p_anfis_->numberOfOutputVariables())
		{
			FL_THROW2(std::invalid_argument, "Incorrect output dimension");
		}

		const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());

		// Compute current rule firing strengths
		const std::vector<fl::scalar> ruleFiringStrengths = p_anfis_->evalTo(entry.inputBegin(), entry.inputEnd(), fl::anfis::Engine::AntecedentLayer);

		// Compute normalization factor
		const fl::scalar totRuleFiringStrength = fl::detail::Sum<fl::scalar>(ruleFiringStrengths.begin(), ruleFiringStrengths.end());

		// Compute input to RLS algorithm
		std::vector<fl::scalar> rlsInputs(rls_.getInputDimension());
		{
			std::size_t k = 0;
			std::size_t r = 0;
			for (std::size_t v = 0,
							 nv = p_anfis_->numberOfOutputVariables();
				 v < nv;
				 ++v)
			{
				fl::OutputVariable* p_var = p_anfis_->getOutputVariable(v);

				FL_DEBUG_ASSERT( p_var );

				for (std::size_t t = 0,
								 nt = p_var->numberOfTerms();
					 t < nt;
					 ++t)
				{
					fl::Term* p_term = p_var->getTerm(t);

					FL_DEBUG_ASSERT( p_term );

					const std::size_t numParams = detail::GetTermParameters(p_term).size();
					for (std::size_t p = 1; p < numParams; ++p)
					{
						rlsInputs[k] = ruleFiringStrengths[r]*entry.getField(p-1)/totRuleFiringStrength;
						++k;
					}
					rlsInputs[k] = ruleFiringStrengths[r]/totRuleFiringStrength;
					++k;
					++r;
				}
			}
		}

		// Estimate parameters
		std::vector<fl::scalar> actualOut;
		actualOut = rls_.estimate(rlsInputs.begin(), rlsInputs.end(), targetOut.begin(), targetOut.end());

		// Put estimated RLS parameters in the ANFIS model and save RLS regressor vector
		{
			const std::vector< std::vector<fl::scalar> > rlsParamMatrix = rls_.getEstimatedParameters();
	//std::cerr << "PHASE #0 - Estimated RLS params: "; fl::detail::MatrixOutput(std::cerr, rlsParamMatrix); std::cerr << std::endl;//XXX

			std::size_t k = 0;
			//std::size_t r = 0;
			for (std::size_t v = 0,
							 nv = p_anfis_->numberOfOutputVariables();
				 v < nv;
				 ++v)
			{
				fl::OutputVariable* p_var = p_anfis_->getOutputVariable(v);

				FL_DEBUG_ASSERT( p_var );

				for (std::size_t t = 0,
								 nt = p_var->numberOfTerms();
					 t < nt;
					 ++t)
				{
					fl::Term* p_term = p_var->getTerm(t);

					FL_DEBUG_ASSERT( p_term );

					const std::size_t numParams = detail::GetTermParameters(p_term).size();
					std::vector<fl::scalar> params(numParams);
					for (std::size_t p = 0; p < numParams; ++p)
					{
						params[p] = rlsParamMatrix[k][v];
						++k;
					}
					detail::SetTermParameters(p_term, params.begin(), params.end());
	//std::cerr << "PHASE #0 - Estimated RLS params - Output #" << v << " - Term #" << t << " - Params: "; fl::detail::VectorOutput(std::cerr, params); std::cerr << std::endl;//XXX
					//++r;
				}
			}

			//rlsPhi_ = rls_.getRegressor();
		}

		// Compute ANFIS output
		actualOut = p_anfis_->eval(entry.inputBegin(), entry.inputEnd());

		// Update bias in case of zero rule firing strength
		if (p_anfis_->hasBias())
		{
			bool skip = false;

			for (std::size_t i = 0,
							 ni = actualOut.size();
				 i < ni;
				 ++i)
			{
				if (fl::Operation::isNaN(actualOut[i]))
				{
					OutputNode* p_outNode = p_anfis_->getOutputLayer().at(i);

					FL_DEBUG_ASSERT( p_outNode );

					//bias_[i] += stepSize_*(targetOut[i]-bias_[i]);
					fl::scalar bias = p_outNode->getBias();
					bias += stepSize_*(targetOut[i]-bias);
					p_outNode->setBias(bias);
					skip = true;
				}
			}
			//p_anfis_->setBias(bias_);

			if (skip)
			{
				// Skip this data point
//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, bias_); std::cerr << std::endl; //XXX
				continue;
			}
		}

//std::cerr << "PHASE #1 - Target output: "; fl::detail::VectorOutput(std::cerr, targetOut); std::cerr << " - ANFIS output: "; fl::detail::VectorOutput(std::cerr, actualOut); std::cerr << " - Bias: "; fl::detail::VectorOutput(std::cerr, bias_); std::cerr << std::endl; //XXX

		// Update error
		fl::scalar se = 0;
		for (std::size_t i = 0,
						 ni = targetOut.size();
			 i < ni;
			 ++i)
		{
			const fl::scalar out = fl::Operation::isNaN(actualOut[i]) ? 0.0 : actualOut[i];

			se += fl::detail::Sqr(targetOut[i]-out);
		}
		rmse += se;
//std::cerr << "PHASE #1 - Current error: " <<  se << " - Total error: " << rmse << std::endl;//XXX

		// Backward errors
		std::map<const Node*,fl::scalar> dEdOs;
		// Computes error derivatives at output layer
		{
			const std::vector<OutputNode*> outLayer = p_anfis_->getOutputLayer();
			for (std::size_t i = 0,
							 ni = targetOut.size();
				 i < ni;
				 ++i)
			{
				const Node* p_node = outLayer[i];

				dEdOs[p_node] = -2.0*(targetOut[i]-actualOut[i]);
//std::cerr << "PHASE #1 - Layer: " << Engine::OutputLayer << ", Node: " << i << ", dEdO: " << dEdOs[p_node] << std::endl;//XXX
			}
		}
		// Propagates errors back to the fuzzification layer
		for (Engine::LayerCategory layerCat = p_anfis_->getPreviousLayerCategory(Engine::OutputLayer);
			 layerCat != Engine::InputLayer;
			 layerCat = p_anfis_->getPreviousLayerCategory(layerCat))
		{
			std::vector<Node*> layer = p_anfis_->getLayer(layerCat);

			for (std::size_t i = 0,
							 ni = layer.size();
				 i < ni;
				 ++i)
			{
				Node* p_fromNode = layer[i];

				fl::scalar dEdO = 0;
				std::vector<Node*> outConns = p_fromNode->outputConnections();
				for (std::size_t j = 0,
								 nj = outConns.size();
					 j < nj;
					 ++j)
				{
					Node* p_toNode = outConns[j];

					const std::vector<fl::scalar> dOdOs = p_toNode->evalDerivativeWrtInputs();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - dOdOs: "; fl::detail::VectorOutput(std::cerr, dOdOs); std::cerr << std::endl;//XXX

					// Find the index k in the input connection of p_fromNode related to the input node p_toNode
					const std::vector<Node*> inConns = p_toNode->inputConnections();
					const std::size_t nk = inConns.size();
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", From Node: " << i << " (" << p_fromNode << ") - To Node: " << j << " (" << p_toNode << ") - #In Conns: " << nk << " - double check #In Conns: " << p_anfis_->inputConnections(p_toNode).size() << std::endl;//XXX
					std::size_t k = 0;
					while (k < nk && inConns[k] != p_fromNode)
					{
						++k;
					}
					if (k == nk)
					{
						FL_THROW2(std::runtime_error, "Found inconsistencies in input and output connections");
					}
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", Node: " << i << " - Found k: " << k << std::endl;//XXX

					dEdO += dEdOs[p_toNode]*dOdOs[k];
				}
//std::cerr << "PHASE #1 - Computing dEdO for Layer: " << layerCat << ", Node: " << i << "..." << std::endl;//XXX
				dEdOs[p_fromNode] = dEdO;
//std::cerr << "PHASE #1 - Layer: " << layerCat << ", Node: " << i << ", dEdO: " << dEdOs[p_fromNode] << std::endl;//XXX
			}
		}

//std::cerr << "PHASE #1 - Updating parameters" << std::endl;
		// Update error derivatives wrt parameters $\frac{\partial E}{\partial P_{ij}}$
		std::vector<FuzzificationNode*> fuzzyLayer = p_anfis_->getFuzzificationLayer();
		for (std::size_t i = 0,
						 ni = fuzzyLayer.size();
			 i < ni;
			 ++i)
		{
			FuzzificationNode* p_node = fuzzyLayer[i];

			const std::vector<fl::scalar> dOdPs = p_node->evalDerivativeWrtParams();
//std::cerr << "PHASE #1 - Layer: " << Engine::FuzzificationLayer << ", Node: " << i << " (" << p_node << "), dOdPs: "; fl::detail::VectorOutput(std::cerr, dOdPs); std::cerr << std::endl;//XXX
			const std::size_t np = dOdPs.size();

			if (dEdPs_.count(p_node) == 0)
			{
				dEdPs_[p_node].resize(np, 0);
			}

			for (std::size_t p = 0; p < np; ++p)
			{
				dEdPs_[p_node][p] += dEdOs[p_node]*dOdPs[p];
//std::cerr << "PHASE #1 - Layer: fuzzification, Node: " << i << " (" << p_node << "), dEdP_" << p << ": " << dEdPs_.at(p_node)[p] << std::endl;//XXX
			}
		}
	}

	rmse = std::sqrt(rmse/data.size());

/*
	if (stepSizeErrWindow_.size() == stepSizeErrWindowLen_)
	{
		stepSizeErrWindow_.pop_back();
		//stepSizeErrWindow_.pop_front();
	}
	stepSizeErrWindow_.push_front(rmse);
	//stepSizeErrWindow_.push_back(rmse);

	p_anfis_->setIsLearning(false);
*/

	return rmse;
}

void Jang1993HybridLearningAlgorithm::reset()
{
	this->init();
}

void Jang1993HybridLearningAlgorithm::init()
{
	std::size_t numParams = 0;
	std::size_t numOutVars = 0;
	if (p_anfis_)
	{
		for (std::size_t i = 0,
						 nv = p_anfis_->numberOfOutputVariables();
			 i < nv;
			 ++i)
		{
			fl::OutputVariable* p_var = p_anfis_->getOutputVariable(i);

			FL_DEBUG_ASSERT( p_var );

			for (std::size_t j = 0,
							 nt = p_var->numberOfTerms();
				 j < nt;
				 ++j)
			{
				fl::Term* p_term = p_var->getTerm(j);

				FL_DEBUG_ASSERT( p_term );

				numParams += detail::GetTermParameters(p_term).size();
			}
		}
		numOutVars = p_anfis_->numberOfOutputVariables();
	}

	rls_.setModelOrder(0);
	rls_.setInputDimension(numParams);
	rls_.setOutputDimension(numOutVars);
	rls_.reset();
	//rlsPhi_.clear();

	dEdPs_.clear();
	stepSize_ = stepSizeInit_;
	stepSizeIncrCounter_ = stepSizeDecrCounter_ = 0;
	stepSizeErrWindow_.clear();

	//bias_.clear();
	//if (useBias_)
	//{
	//	bias_.resize(p_anfis_->numberOfOutputVariables(), 0);
	//}
}

void Jang1993HybridLearningAlgorithm::check()
{
	if (p_anfis_ == fl::null)
	{
		FL_THROW2(std::logic_error, "Invalid ANFIS engine");
	}
	if (stepSizeInit_ <= 0)
	{
		FL_THROW2(std::logic_error, "Invalid step-size");
	}
	if (stepSizeDecrRate_ <= 0)
	{
		FL_THROW2(std::logic_error, "Invalid step-size decreasing rate");
	}
	if (stepSizeIncrRate_ <= 0)
	{
		FL_THROW2(std::logic_error, "Invalid step-size increasing rate");
	}
	if (stepSizeErrWindowLen_ == 0)
	{
		FL_THROW2(std::logic_error, "Invalid length for the step-size error window");
	}
}

}} // Namespace fl::anfis
