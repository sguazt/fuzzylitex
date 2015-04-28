/**
 * \file anfis_mackeyglass.cpp
 *
 * Based on the MATLAB Fuzzy Logic Toolbox example:
 *  "Predict Chaotic Time-Series" (see [1,2]).
 * The following is a description of the problem as found in [1,2].
 *
 * This example uses an ANFIS model to predict a time series that is generated
 * by the following Mackey-Glass (MG) time-delay differential equation:
 * \f[
 *  \dot{x}(t) = \frac{0.2x(t-\tau)}{1+x^{10}(t-\tau)}=0.1x(t)
 * \f]
 * This time series is chaotic with no clearly defined period.
 * The series does not converge or diverge, and the trajectory is highly
 * sensitive to initial conditions.
 * This benchmark problem is used in the neural network and fuzzy modeling
 * research communities (e.g., see [3,4]).
 *
 * References
 * -# The MathWorks, Inc., "Predict Chaotic Time-Series with MATLAB Fuzzy Logic Toolbox," 2015,
 *    URL: http://www.mathworks.com/help/fuzzy/predict-chaotic-time-series-code.html
 * -# The MathWorks, Inc., "Chaotic Time-Series Prediction with MATLAB Fuzzy Logic Toolbox," 2015,
 *    URL: http://www.mathworks.com/help/fuzzy/examples/chaotic-time-series-prediction.html
 * -# J.-S.R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference Systems," IEEE Transactions on Systems, Man, and Cybernetics, 23:3(665-685), 1993.
 * -# D. Nauck and R. Kruse, "A Neuro-Fuzzy Approach to Obtain Interpretable Fuzzy Systems for Function Approximation," In Proc. of the IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), 1998.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2015 Marco Guazzone (marco.guazzone@gmail.com)
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


#include <cmath>
#include <cstddef>
#include <cstring>
#include <fl/fis_builders.h>
#include <fl/anfis.h>
#include <fl/dataset.h>
#include <fl/Headers.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace /*<unnamed>*/ {

const std::size_t DefaultMaxEpochs = 10;
const fl::scalar DefaultGoalError = 0;
const fl::scalar DefaultMomentum = 0;
const bool DefaultOnlineMode = false;
const fl::scalar DefaultTolerance = 1.0e-10;


/// Checks if the given arguments are equal with respect to the given tolerance
template <typename T>
bool CheckEqual(T x, T y, T tol = DefaultTolerance);

/// Shows a usage message
void usage(const char* progname);

/// Makes training and test datasets
void MakeTrainAndTestSets(fl::DataSet<>& trainingSet, fl::DataSet<>& testSet);

/// Builds an ANFIS model by means of the grid partitioning method according to
/// the given training set
FL_unique_ptr<fl::anfis::Engine> BuildAnfis(const fl::DataSet<>& trainingSet);

/// Trains the given ANFIS model with the given training set and according to
/// the given number of epochs and training error goal
fl::scalar TrainAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& trainingSet, std::size_t maxEpochs, fl::scalar goalError, fl::scalar momentum, bool online);

/// Tests the trained ANFIS model against the given test set
fl::scalar TestAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& testSet);

/**
 * Check the trained ANFIS against MATLAB's results
 *
 * This function uses data stored in file 'data/mgdata_check.dat'.
 * Each row in this data file has two columns, where the first column represents
 * the true output of the time series and the second column represents the
 * output obtained by the ANFIS trained with MATLAB 2012b.
 * The data file contains 1000 rows.
 * The first 500 rows are related to data in the training set, while the latter
 * 500 rows are related to data in the test set.
 */
fl::scalar ValidateAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& trainingSet, const fl::DataSet<>& testSet);


template <typename T>
bool CheckEqual(T x, T y, T tol)
{
	if (std::isnan(x) || std::isnan(y))
	{
		// According to IEEE, NaN are different event by itself
		return false;
	}
	return std::abs(x-y) <= (std::min(std::abs(x), std::abs(y))*tol);
}

void usage(const char* progname)
{
	std::cout << "Usage: " << progname << " [options]" << std::endl
			  << "Options:" << std::endl
			  << "--help: Show this help message." << std::endl
			  << "--epoch <value>: Set the maximum number of epochs in the training algorithm." << std::endl
			  << "  [default: " << DefaultMaxEpochs << "]" << std::endl
			  << "--error <value>: Set the maximum error to achieve in the training algorithm." << std::endl
			  << "  [default: " << DefaultGoalError << "]" << std::endl
			  << std::endl;
}

void MakeTrainAndTestSets(fl::DataSet<>& trainingSet, fl::DataSet<>& testSet)
{
	// To obtain the time series value at integer points, the fourth-order
	// Runge-Kutta method is used to find the numerical solution to the
	// previous MG equation.
	// It is assumed that $x(0)= 1.2$, $\tau=17$, and $x(t)=0$ for $t<0$.
	// The result was saved in the file 'data/mgdata.dat'.
	std::ifstream ifs("data/mgdata.dat");
	std::vector<fl::scalar> x;
	for (std::string line; std::getline(ifs, line); )
	{
		std::istringstream iss(line);
		fl::scalar t = 0;
		iss >> t;
		fl::scalar v = 0;
		iss >> v;
		x.push_back(v);
	}
	ifs.close();

	// In time-series prediction, you need to use known values of the time
	// series up to the point in time, $t$, to predict the value at some point
	// in the future, $t + P$.
	// The standard method for this type of prediction is to create a mapping
	// from $D$ sample data points, sampled every $\Delta$ units in time,
	// ($x(t-(D-1)\Delta),\ldots,x(t-\Delta),x(t)$), to a predicted future value
	// $x=(t+P)$.
	// Following the conventional settings for predicting the MG time series,
	// set $D = 4$ and $\Delta = P = 6$.
	// For each $t$, the input training data for anfis is a four-column vector
	// of the following form:
	// \[
	//  w(t) = [x(t-19), x(t-12), x(t-6), x(t)]
	// \]
	// The output training data corresponds to the trajectory prediction:
	// \[
	//  s(t) = x(t+6)
	// \]
	// For each $t$, ranging in values from 118 to 1117, the training
	// input/output data is a structure whose first component is the
	// four-dimensional input $w$, and whose second component is the output $s$.
	// There are 1000 input/output data values.
	// You use the first 500 data values for the anfis training (these become
	// the training data set), while the others are used as checking data for
	// validating the identified fuzzy model.
	for (std::size_t t = 117; t < 1117; ++t)
	{
		// Setup the input
		std::vector<fl::scalar> xx;
		xx.push_back(x[t-18]);
		xx.push_back(x[t-12]);
		xx.push_back(x[t-6]);
		xx.push_back(x[t]);
		// Setup the output
		std::vector<fl::scalar> yy;
		yy.push_back(x[t+6]);
		// Build the entry
		fl::DataSetEntry<fl::scalar> entry(xx.begin(), xx.end(), yy.begin(), yy.end());
		if (trainingSet.size() < 500)
		{
			trainingSet.add(entry);
		}
		else
		{
			testSet.add(entry);
		}
	}
}

FL_unique_ptr<fl::anfis::Engine> BuildAnfis(const fl::DataSet<>& trainingSet)
{
	// Generates FIS with the grid partitioning method
	std::vector<std::size_t> numMFs;
	numMFs.push_back(2);
	numMFs.push_back(2);
	numMFs.push_back(2);
	numMFs.push_back(2);

	std::vector<std::string> inMFs;
	inMFs.push_back(fl::Bell().className());
	inMFs.push_back(fl::Bell().className());
	inMFs.push_back(fl::Bell().className());
	inMFs.push_back(fl::Bell().className());

	fl::GridPartitionFisBuilder<fl::anfis::Engine> fisBuilder(numMFs.begin(), numMFs.end(), inMFs.begin(), inMFs.end(), fl::Linear().className());
	FL_unique_ptr<fl::anfis::Engine> p_anfis = fisBuilder.build(trainingSet);
	// Builds the ANFIS for the generated FIS
	p_anfis->setName("Mackey-Glass");
	p_anfis->build();

	return p_anfis;
}

fl::scalar TrainAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& trainingSet, std::size_t maxEpochs, fl::scalar goalError, fl::scalar momentum, bool online)
{
	fl::scalar rmse = 0;

	fl::anfis::Jang1993HybridLearningAlgorithm hybridLearner(p_anfis);
	if (online)
	{
		hybridLearner.setIsOnline(true);
		hybridLearner.setForgettingFactor(0.98);
	}
	else
	{
		hybridLearner.setIsOnline(false);
	}
	hybridLearner.setMomentum(momentum);
	rmse = hybridLearner.train(trainingSet, maxEpochs, goalError);

	return rmse;
}

fl::scalar TestAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& testSet)
{
	fl::scalar rmse = 0;

	for (fl::DataSet<fl::scalar>::ConstEntryIterator dataIt = testSet.entryBegin(),
													 dataEndIt = testSet.entryEnd();
		 dataIt != dataEndIt;
		 ++dataIt)
	{
		const fl::DataSetEntry<fl::scalar> entry = *dataIt;
		const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());
		const std::vector<fl::scalar> anfisOut = p_anfis->eval(entry.inputBegin(), entry.inputEnd());

		for (std::size_t i = 0; i < targetOut.size(); ++i)
		{
			const fl::scalar err = targetOut[i]-anfisOut[i];

			rmse += err*err;
		}
	} 
	rmse = std::sqrt(rmse/testSet.size());

	return rmse;
}

fl::scalar ValidateAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& trainingSet, const fl::DataSet<>& testSet)
{
	std::ifstream ifs("data/mgdata_check.dat");

	fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE)

	// - Check prediction over the training set
	for (fl::DataSet<fl::scalar>::ConstEntryIterator dataIt = trainingSet.entryBegin(),
													 dataEndIt = trainingSet.entryEnd();
		 dataIt != dataEndIt;
		 ++dataIt)
	{
		const fl::DataSetEntry<fl::scalar> entry = *dataIt;
		const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());
		const std::vector<fl::scalar> anfisOut = p_anfis->eval(entry.inputBegin(), entry.inputEnd());

		std::string line;
		std::getline(ifs, line);
		std::istringstream iss(line);
		fl::scalar trueOut = 0; // true output
		iss >> trueOut;
		fl::scalar matlabOut = 0; // matlab output
		iss >> matlabOut;

		const fl::scalar err = matlabOut-anfisOut.back();
		rmse += err*err;
	}
	// - Check prediction over the test set
	for (fl::DataSet<fl::scalar>::ConstEntryIterator dataIt = testSet.entryBegin(),
													 dataEndIt = testSet.entryEnd();
		 dataIt != dataEndIt;
		 ++dataIt)
	{
		const fl::DataSetEntry<fl::scalar> entry = *dataIt;
		const std::vector<fl::scalar> targetOut(entry.outputBegin(), entry.outputEnd());
		const std::vector<fl::scalar> anfisOut = p_anfis->eval(entry.inputBegin(), entry.inputEnd());

		std::string line;
		std::getline(ifs, line);
		std::istringstream iss(line);
		fl::scalar trueOut = 0; // true output
		iss >> trueOut;
		fl::scalar matlabOut = 0; // matlab output
		iss >> matlabOut;

		const fl::scalar err = matlabOut-anfisOut.back();

		if (!CheckEqual(matlabOut, anfisOut.back()))
		{
			std::cerr << "[Warning] MATLAB: " << matlabOut << " vs. FL: " << anfisOut.back() << " -> Abs Error: " << std::abs(err) << std::endl;
		}

		rmse += err*err;
	}
	rmse = std::sqrt(rmse/(trainingSet.size()+testSet.size()));

	ifs.close();

	return rmse;
}

} // Namespace <unnamed>


int main(int argc, char* argv[])
{
	std::size_t maxEpochs = DefaultMaxEpochs;
	fl::scalar goalError = DefaultGoalError;
	fl::scalar momentum = DefaultMomentum;
	bool online = DefaultOnlineMode;

	for (int i = 1; i < argc; ++i)
	{
		if (!std::strcmp(argv[i], "--help"))
		{
			usage(argv[0]);
			break;
		}
		else if (!std::strcmp(argv[i], "--epoch"))
		{
			++i;
			if (i < argc)
			{
				std::istringstream iss(argv[i]);
				iss >> maxEpochs;
			}
		}
		else if (!std::strcmp(argv[i], "--error"))
		{
			++i;
			if (i < argc)
			{
				std::istringstream iss(argv[i]);
				iss >> goalError;
			}
		}
		else if (!std::strcmp(argv[i], "--momentum"))
		{
			++i;
			if (i < argc)
			{
				std::istringstream iss(argv[i]);
				iss >> momentum;
			}
		}
		else if (!std::strcmp(argv[i], "--offline"))
		{
			online = false;
		}
		else if (!std::strcmp(argv[i], "--online"))
		{
			online = true;
		}
	}

	std::cout << "Options:" << std::endl
			  << "- max epochs: " << maxEpochs << std::endl
			  << "- error goal: " << goalError  << std::endl
			  << "- online: " << std::boolalpha << online << std::endl
			  << "- momentum: " << momentum << std::endl
			  << std::endl;

	//fl::fuzzylite::setDebug(true);

	// Creates training set and test set

	std::cout << "Making datasets..." << std::endl;

	fl::DataSet<fl::scalar> trainingSet(4,1);
	fl::DataSet<fl::scalar> testSet(4,1);

	MakeTrainAndTestSets(trainingSet, testSet);

	std::cout << "Building..." << std::endl;

	FL_unique_ptr<fl::anfis::Engine> p_anfis = BuildAnfis(trainingSet);

	std::cout << "Built:" << std::endl << p_anfis->toString() << std::endl;

	fl::scalar rmse = 0; // The Root Mean Squared Error (RMSE)

	std::cout << "Training..." << std::endl;

	rmse = TrainAnfis(p_anfis.get(), trainingSet, maxEpochs, goalError, momentum, online);

	std::cout << "Trained -> RMSE: " << rmse << std::endl;
	std::cout << "Trained -> FIS: " << std::endl
			  << p_anfis->toString() << std::endl;

	std::cout << "Testing..." << std::endl;

	rmse = TestAnfis(p_anfis.get(), testSet);

	std::cout << "Tested -> RMSE: " << rmse << std::endl;

	std::cout << "Validating..." << std::endl;

	rmse = ValidateAnfis(p_anfis.get(), trainingSet, testSet);

	std::cout << "Validating -> RMSE: " << rmse << std::endl;
}
