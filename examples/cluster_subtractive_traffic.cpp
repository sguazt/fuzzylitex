/**
 * \file cluster_subtractive_tripdata.cpp
 *
 * Based on the MATLAB Fuzzy Logic Toolbox example:
 *  "Modeling Traffic Patterns using Subtractive Clustering"
 * See [1], in the references below.
 * The following is a description of the problem as found in [1].
 *
 * The purpose of this example is to understand the relationship between the
 * number of automobile trips generated from an area and the area's
 * demographics.
 * Demographic and trip data were collected from traffic analysis zones in New
 * Castle County, Delaware.
 * Five demographic factors are considered: population, number of dwelling units, vehicle ownership, median household income and total employment.
 *
 * Hereon, the demographic factors will be addressed as inputs and the trips
 * generated will be addressed as output.
 * Hence the problem has five input variables (five demographic factors) and one
 * output variable (number of trips generated).
 *
 * The relationship between the input variables (demographics) and the output
 * variable (trips) is modeled by clustering the data.
 *
 * References
 * -# The Mathworks, Inc., "Modeling Traffic Patterns using Subtractive Clustering", 2015,
 *    URL: http://it.mathworks.com/help/fuzzy/examples/modeling-traffic-patterns-using-subtractive-clustering.html
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


#include <cstddef>
#include <fl/cluster/subtractive.h>
#include <fl/dataset.h>
#include <fl/fis_builder/subtractive_clustering.h>
#include <fl/fuzzylite.h>
#include <fl/Engine.h>
#include <fl/macro.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


namespace /*<unnamed>*/ {

void MakeDataSet(fl::DataSet<>& data);

void MakeDataSet(fl::DataSet<>& data)
{
	const std::size_t ni = 5;
	const std::size_t no = 1;

	data = fl::DataSet<>(ni, no);

	std::vector<std::string> labels;
	labels.push_back("population");
	labels.push_back("num. of dwelling units");
	labels.push_back("vehicle ownership");
	labels.push_back("median household income");
	labels.push_back("total employment");
	labels.push_back("num. of trips");

	FL_DEBUG_ASSERT( labels.size() == (ni+no) );

	std::ifstream ifs("data/tripdata.dat");
	data.setLabels(labels.begin(), labels.end());
	for (std::string line; std::getline(ifs, line); )
	{
		std::vector<fl::scalar> in;
		std::vector<fl::scalar> out;

		std::istringstream iss(line);
		for (std::size_t i = 0; iss.good() && i < no; ++i)
		{
			fl::scalar x;
			iss >> x;
			out.push_back(x);
//std::cerr << "OUT #" << i << " -> " << x << std::endl;
		}
		for (std::size_t i = 0; iss.good() && i < ni; ++i)
		{
			fl::scalar x;
			iss >> x;
			in.push_back(x);
//std::cerr << "IN #" << i << " -> " << x << std::endl;
		}

		FL_DEBUG_ASSERT( in.size() == ni );
		FL_DEBUG_ASSERT( out.size() == no );

		fl::DataSetEntry<fl::scalar> entry(in.begin(), in.end(), out.begin(), out.end());
		data.add(entry);
	}
}

} // Namespace <unnamed>

int main()
{
	fl::DataSet<> dataset;

	MakeDataSet(dataset);

	fl::cluster::SubtractiveClustering subclust;
	subclust.cluster(dataset.data());
	const std::vector< std::vector<fl::scalar> > centers = subclust.centers();
	const std::vector<fl::scalar> sigma = subclust.rangeOfInfluence();

	std::cout << "Centers:" << std::endl;;
	for (std::size_t i = 0,
					 ni = centers.size();
		 i < ni;
		 ++i)
	{
		std::cout << "Cluster #" << (i+1) << "[";
		for (std::size_t j = 0,
						 nj = centers[i].size();
			 j < nj;
			 ++j)
		{
			if (j > 0)
			{
				std::cout << " ";
			}
			std::cout << centers[i][j];
		}
		std::cout << "]" << std::endl;
	}
	std::cout << "Range of Influence: [";
	for (std::size_t i = 0,
					 ni = sigma.size();
		 i < ni;
		 ++i)
	{
		if (i > 0)
		{
			std::cout << " ";
		}
		std::cout << sigma[i];
	}
	std::cout << "]" << std::endl;

/*
C =

    1.8770    0.7630    0.9170   18.7500    1.5650    2.1830
    0.3980    0.1510    0.1320    8.1590    0.6250    0.6480
    3.1160    1.1930    1.4870   19.7330    0.6030    2.3850


S =

    1.1621    0.4117    0.6555    7.6139    2.8931    1.4395
*/

	fl::SubtractiveClusteringFisBuilder<fl::Engine> fisBuilder(subclust);
	FL_unique_ptr<fl::Engine> p_fis = fisBuilder.build(dataset);
	std::cout << "Built FIS: " << p_fis->toString() << std::endl;
}
