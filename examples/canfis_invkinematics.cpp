/**
 * \file canfis_invkinematics.cpp
 *
 * \brief Modeling inverse kinematics in two planar robotic arms
 *
 * Based on the section 19.3:
 *  "Inverse Kinematics Problems"
 * of [1] and on the MATLAB Fuzzy Logic Toolbox example:
 *  "Modeling Inverse Kinematics in a Robotic Arm" (see [2]).
 * The following is a description of the problem as found in [1,2].
 *
 * Kinematics is the science of motion.
 * In a two-joint robotic arm, given the angles \f$\theta_1\f$ and
 * \f$\theta_2\f$ of the joints, the kinematics equations give the location
 * \f$(x,y)\f$ of the tip of the arm:
 * \f{align}
 *  x &= l_1\cos(\theta_1)+l_2\cos(\theta_1+\theta_2),\\
 *  y &= l_1\sin(\theta_1)+l_2\sin(\theta_1+\theta_2).
 * \f}
 * where \f$l_1\f$ and \f$l_2\f$ are the arm lengths, and \f$\theta_1\f$ and
 * \f$\theta_2\f$ their respective angles.
 *
 * Inverse kinematics refers to the reverse process.
 * Given a desired location for the tip of the robotic arm, what should the
 * angles \f$\theta_1\f$ and \f$\theta_2\f$ of the joints be so as to locate
 * the tip of the arm at the desired location \f$(x,y)\f$.
 * There is usually more than one solution and can at times be a difficult
 * problem to solve.
 *
 * This is a typical problem in robotics that needs to be solved to control a
 * robotic arm to perform tasks it is designated to do.
 * In a 2-dimensional input space, with a two-joint robotic arm and given the
 * desired coordinate \f$(x,y)\f$, the problem reduces to finding the two angles
 * \f$\theta_1\f$ and \f$\theta_2\f$ involved.
 * The first angle \f$\theta_1\f$ is between the first arm and the ground (or
 * whatever it is attached to).
 * The second angle \f$\theta_2\f$ is between the first arm and the second arm.
 *
 * The modeled system has thus two inputs, namely the \f$(x,y)\f$ coordinate,
 * and two outputs, namely the \f$\theta_1\f$ and \f$\theta_2\f$ angles.
 *
 * One approach to building an ANFIS solution for this problem, is to build two
 * ANFIS networks (i.e., to use a MANFIS architecture), one to predict
 * \f$\theta_1\f$ and the other to predict \f$\theta_2\f$.
 *
 * References:
 * -# J.-S.R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
 * -# The MathWorks, Inc., "Modeling Inverse Kinematics in a Robotic Arm with MATLAB Fuzzy Logic Toolbox," 2015,
 *    URL: http://www.mathworks.com/help/fuzzy/examples/modeling-inverse-kinematics-in-a-robotic-arm.html
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
#include <fl/anfis.h>
#include <fl/dataset.h>
#include <fl/fis_builders.h>
#include <fl/Headers.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace /*<unnamed>*/ {

const std::size_t DefaultMaxEpochs = 150;
const fl::scalar DefaultGoalError = 0;
const bool DefaultOnlineMode = false;
const fl::scalar DefaultTolerance = 1e-7;

const fl::scalar l1 = 10; // Length of first arm
const fl::scalar l2 = 7; // Length of second arm


/// Checks if the given arguments are equal with respect to the given tolerance
template <typename T>
bool CheckEqual(T x, T y, T tol = DefaultTolerance);

/// Shows a usage message
void usage(const char* progname);

/// Prints the given vector on the given stream
template <typename C, typename CT, typename T>
void PrintVector(std::basic_ostream<C,CT>& os, const std::vector<T>& v);

template <typename C, typename CT, typename T>
void PrintMatrix(std::basic_ostream<C,CT>& os, const std::vector< std::vector<T> >& M);

template <typename C, typename CT, typename T>
void PrintDataSet(std::basic_ostream<C,CT>& os, const fl::DataSet<T>& data);

void MakeTrainingSet(fl::DataSet<>& data);

void MakeTestSet(fl::DataSet<>& data);

FL_unique_ptr<fl::anfis::Engine> BuildAnfis(const fl::DataSet<>& data, std::size_t numInMFs);

fl::scalar TrainAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& data, std::size_t maxEpochs, fl::scalar goalError, bool online);

fl::scalar TestAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& testSet);


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
			  << "--online: Set the training algorithm to operate in online mode." << std::endl
			  << "  [default: " << DefaultOnlineMode << "]" << std::endl
			  << "--offline: Set the training algorithm to operate in offline mode." << std::endl
			  << "  [default: " << !DefaultOnlineMode << "]" << std::endl
			  << std::endl;
}

template <typename C, typename CT, typename T>
void PrintVector(std::basic_ostream<C,CT>& os, const std::vector<T>& v)
{
	os << "[";
	for (std::size_t i = 0,
					 ni = v.size();
		 i < ni;
		 ++i)
	{
		os << v[i] << " ";
	}
	os << "]";
}

template <typename C, typename CT, typename T>
void PrintMatrix(std::basic_ostream<C,CT>& os, const std::vector< std::vector<T> >& M)
{
	os << "[";
	for (std::size_t i = 0,
					 ni = M.size();
		 i < ni;
		 ++i)
	{
		PrintVector(M[i]);
		os << "; ";
	}
	os << "]";
}

template <typename C, typename CT, typename T>
void PrintDataSet(std::basic_ostream<C,CT>& os, const fl::DataSet<T>& data)
{
	os << "[";
	for (typename fl::DataSet<>::ConstEntryIterator it = data.entryBegin(),
													endIt = data.entryEnd();
		 it != endIt;
		 ++it)
	{
		const fl::DataSetEntry<>& entry = *it;

		for (typename fl::DataSetEntry<>::ConstInputIterator inIt = entry.inputBegin(),
															 inEndIt = entry.inputEnd();
			 inIt != inEndIt;
			 ++inIt)
		{
			os << *inIt << " ";
		}
		for (typename fl::DataSetEntry<>::ConstOutputIterator outIt = entry.outputBegin(),
															  outEndIt = entry.outputEnd();
			 outIt != outEndIt;
			 ++outIt)
		{
			os << *outIt << " ";
		}
		os << std::endl;
	}
	os << "]";
}

void MakeTrainingSet(fl::DataSet<>& data)
{
	const std::size_t numInputs = 2;
	const fl::scalar step = 0.1;
	const fl::scalar pi = static_cast<fl::scalar>(3.1415926535897932384626433832795029L);
	const fl::scalar halfPi = pi/2.0;

	// {MATLAB: theta1 = [0.0:0.1:pi/2]}
	std::vector<fl::scalar> theta1;
	for (fl::scalar seq = 0; seq <= halfPi; seq += step)
	{
		theta1.push_back(seq);
	}
	// {MATLAB: theta2 = [0.0:0.1:pi]}
	std::vector<fl::scalar> theta2;
	for (fl::scalar seq = 0; seq <= pi; seq += step)
	{
		theta2.push_back(seq);
	}
	// {MATLAB: [THETA1, THETA2] = meshgrid(theta1, theta2)}
	// - THETA1: each row is a duplicate of theta1
	std::vector< std::vector<fl::scalar> > THETA1(theta2.size());
	for (std::size_t i = 0,
					 ni = theta2.size();
		 i < ni;
		 ++i)
	{
		THETA1[i] = theta1;
	}
	// - THETA2: each column is a duplicate of theta2
	std::vector< std::vector<fl::scalar> > THETA2(theta2.size());
	for (std::size_t i = 0,
					 ni = theta2.size();
		 i < ni;
		 ++i)
	{
		THETA2[i] = std::vector<fl::scalar>(theta1.size(), theta2[i]);
	}
	//std::cout << "THETA1: "; PrintMatrix(std::cout, THETA1); std::cout << std::endl;
	//std::cout << "THETA2: "; PrintMatrix(std::cout, THETA2); std::cout << std::endl;

	// Computes x and y coordinates
	// {MATLAB: X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2);}
	// {MATLAB: Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2);}
	std::vector< std::vector<fl::scalar> > X(THETA1.size());
	std::vector< std::vector<fl::scalar> > Y(THETA1.size());
	for (std::size_t i = 0,
					 ni = THETA1.size();
		 i < ni;
		 ++i)
	{
		X[i].resize(THETA1[i].size());
		Y[i].resize(THETA1[i].size());

		for (std::size_t j = 0,
						 nj = THETA1[i].size();
			 j < nj;
			 ++j)
		{
			// Computes x coordinates
			X[i][j] = l1*std::cos(THETA1[i][j]) + l2*std::cos(THETA1[i][j]+THETA2[i][j]);
			// Computes y coordinates
			Y[i][j] = l1*std::sin(THETA1[i][j]) + l2*std::sin(THETA1[i][j]+THETA2[i][j]);
		}
	}
	//std::cout << "X: "; PrintMatrix(std::cout, X); std::cout << std::endl;
	//std::cout << "Y: "; PrintMatrix(std::cout, Y); std::cout << std::endl;

	// Creates x-y-theta1-theta2 dataset
	// {MATLAB: data = [X(:) Y(:) THETA1(:) THETA2(:)];}
	data = fl::DataSet<>(numInputs, 2);
	for (std::size_t j = 0,
					 nj = (THETA1.size() > 0) ? THETA1[0].size() : 0u;
		 j < nj;
		 ++j)
	{
		for (std::size_t i = 0,
						 ni = THETA1.size();
			 i < ni;
			 ++i)
		{
			fl::DataSetEntry<> entry;

			std::vector<fl::scalar> inputs(numInputs);
			inputs[0] = X[i][j];
			inputs[1] = Y[i][j];
			entry.setInputs(inputs.begin(), inputs.end());

			std::vector<fl::scalar> outputs(2);
			outputs[0] = THETA1[i][j];
			outputs[1] = THETA2[i][j];
			entry.setOutputs(outputs.begin(), outputs.end());
			data.add(entry);
		}
	}
}

void MakeTestSet(fl::DataSet<>& data)
{
	const std::size_t numInputs = 2;

	// x coordinates for validation
	// {MATLAB: x = 0:0.1:2;}
	std::vector<fl::scalar> x;
	for (fl::scalar seq = 0; seq <= (2.0+1e-9); seq += 0.1)
	{
		x.push_back(seq);
	}
	// y coordinates for validation
	// {MATLAB: y = 8:0.1:10;}
	std::vector<fl::scalar> y;
	for (fl::scalar seq = 8; seq <= (10.0+1e-9); seq += 0.1)
	{
		y.push_back(seq);
	}

	// Combines x and y coordinates
	// {MATLAB: [X, Y] = meshgrid(x, y);}
	// - X: each row is a duplicate of x
	std::vector< std::vector<fl::scalar> > X(y.size());
	for (std::size_t i = 0,
					 ni = y.size();
		 i < ni;
		 ++i)
	{
		X[i] = x;
	}
	// - Y: each column is a duplicate of y
	std::vector< std::vector<fl::scalar> > Y(y.size());
	for (std::size_t i = 0,
					 ni = y.size();
		 i < ni;
		 ++i)
	{
		Y[i] = std::vector<fl::scalar>(x.size(), y[i]);
	}

	// {MATLAB: c2 = (X.^2 + Y.^2 - l1^2 - l2^2)/(2*l1*l2);}
	// {MATLAB: s2 = sqrt(1 - c2.^2);}
	// {MATLAB: THETA2D = atan2(s2, c2);}
	// {MATLAB: k1 = l1 + l2.*c2;}
	// {MATLAB: k2 = l2*s2;}
	// {MATLAB: THETA1D = atan2(Y, X) - atan2(k2, k1);}
	std::vector< std::vector<fl::scalar> > THETA1D(X.size());
	std::vector< std::vector<fl::scalar> > THETA2D(X.size());
	for (std::size_t i = 0,
					 ni = X.size();
		 i < ni;
		 ++i)
	{
		THETA1D[i].resize(X[i].size());
		THETA2D[i].resize(X[i].size());

		for (std::size_t j = 0,
						 nj = X[i].size();
			 j < nj;
			 ++j)
		{
			const fl::scalar c2 = (X[i][j]*X[i][j] + Y[i][j]*Y[i][j] - l1*l1 -l2*l2)/(2.0*l1*l2);
			const fl::scalar s2 = std::sqrt(1-c2*c2);
			THETA2D[i][j] = std::atan2(s2, c2); // theta2 is deduced

			const fl::scalar k1 = l1 + l2*c2;
			const fl::scalar k2 = l2*s2;
			THETA1D[i][j] = std::atan2(Y[i][j], X[i][j]) - std::atan2(k2, k1); // theta1 is deduced
		}
	}

	// {MATLAB: XY1 = [X(:) Y(:) THETA1D THETA2D];}
	data = fl::DataSet<>(numInputs, 2);
	for (std::size_t j = 0,
					 nj = (THETA1D.size() > 0) ? THETA1D[0].size() : 0u;
		 j < nj;
		 ++j)
	{
		for (std::size_t i = 0,
						 ni = THETA1D.size();
			 i < ni;
			 ++i)
		{
			fl::DataSetEntry<> entry;

			std::vector<fl::scalar> inputs(numInputs);
			inputs[0] = X[i][j];
			inputs[1] = Y[i][j];
			entry.setInputs(inputs.begin(), inputs.end());

			std::vector<fl::scalar> outputs(2);
			outputs[0] = THETA1D[i][j];
			outputs[1] = THETA2D[i][j];
			entry.setOutputs(outputs.begin(), outputs.end());
			data.add(entry);
		}
	}
}

FL_unique_ptr<fl::anfis::Engine> BuildAnfis(const fl::DataSet<>& data, std::size_t numInMFs)
{
	//  {MATLAB: fis = genfis(data, numInMFs);}

	const std::size_t numInputs = 2;

	// Generate rules with the grid partition method
	std::vector<std::size_t> numMFs(numInputs, numInMFs);
	std::vector<std::string> inMFs(numInputs, fl::Bell().className());
	fl::GridPartitionFisBuilder<fl::anfis::Engine> fisBuilder(numMFs.begin(), numMFs.end(), inMFs.begin(), inMFs.end(), fl::Linear().className());
	FL_unique_ptr<fl::anfis::Engine> p_anfis = fisBuilder.build(data);

	// Build the ANFIS
	p_anfis->build();

	return p_anfis;
}

fl::scalar TrainAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& data, std::size_t maxEpochs, fl::scalar goalError, bool online)
{
	//  {MATLAB: fis2 = anfis(fis, maxEpochs, [0,0,0,0]);}

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
	return hybridLearner.train(data, maxEpochs, goalError);
}

fl::scalar TestAnfis(fl::anfis::Engine* p_anfis, const fl::DataSet<>& testSet)
{
	// {MATLAB: x = 0:0.1:2;}
	// {MATLAB: y = 8:0.1:10;}
	// ...
	// {MATLAB: XY = [X(:) Y(:)];}
	// {MATLAB: THETAP = evalfis(XY, anfis);}
	// {MATLAB: thetadiff = THETAD(:) - THETAP;}

	fl::scalar rmse = 0;
	for (fl::DataSet<>::ConstEntryIterator entryIt = testSet.entryBegin(),
										   entryEndIt = testSet.entryEnd();
		 entryIt != entryEndIt;
		 ++entryIt)
	{
		const fl::DataSetEntry<> entry = *entryIt;

		const std::vector<fl::scalar> THETAD(entry.outputBegin(), entry.outputEnd());
		const std::vector<fl::scalar> THETAP = p_anfis->eval(entry.inputBegin(), entry.inputEnd());

		for (std::size_t i = 0,
						 ni = THETAD.size();
			 i < ni;
			 ++i)
		{
			const fl::scalar err = THETAD[i]-THETAP[i];
			rmse += err*err;
		}
	}
	rmse = std::sqrt(rmse/testSet.size());

	return rmse;
}

} // Namespace <unnamed>


int main(int argc, char* argv[])
{
	std::size_t maxEpochs = DefaultMaxEpochs;
	fl::scalar goalError = DefaultGoalError;
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
			  << std::endl;

	//fl::fuzzylite::setDebug(true);

	// Creates training sets

	std::cout << "Making training set..." << std::endl;

	fl::DataSet<> data; // x-y-theta1-theta2 dataset

	MakeTrainingSet(data);

	std::cout << "Data1: "; PrintDataSet(std::cout, data); std::cout << std::endl;

	// Building the ANFIS models

	std::cout << "Building the ANFIS network..." << std::endl;
	//  MATLAB: anfis = anfis(data, 7, 150, [0,0,0,0])
	FL_unique_ptr<fl::anfis::Engine> p_anfis;
	p_anfis = BuildAnfis(data, 7);
	p_anfis->setName("ANFIS");
	std::cout << "Built: " << std::endl
			  << p_anfis->toString() << std::endl;

	// Training the ANFIS models

	fl::scalar rmse = 0;

	std::cout << "Training the ANFIS network..." << std::endl;
	rmse = TrainAnfis(p_anfis.get(), data, maxEpochs, goalError, online);
	std::cout << "Trained -> RMSE: " << rmse << std::endl;
	std::cout << "Trained -> MODEL: " << std::endl
			  << p_anfis->toString() << std::endl;


	// Testing the ANFIS models

	std::cout << "Making test sets..." << std::endl;

	fl::DataSet<> testData;
	MakeTestSet(testData);

	//std::cout << "Test Data: "; PrintDataSet(std::cout, testData); std::cout << std::endl;

	std::cout << "Testing ANFIS network..." << std::endl;

	std::cout << "Testing the ANFIS network..." << std::endl;
	rmse = TestAnfis(p_anfis.get(), testData);
	std::cout << "Tested -> RMSE: " << rmse << std::endl;
}
