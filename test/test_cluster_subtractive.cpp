/**
 * \file test/test_cluster_subtractive.cpp
 *
 * \brief Test suite for subtractive clustering.
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
#include <fl/detail/traits.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>


namespace /*<unnnamed>*/ {

namespace detail {

const double DefaultTolerance = std::numeric_limits<double>::epsilon()*100.0;

void Setup(fl::cluster::SubtractiveClustering& subclust)
{
	// The Trip Generation Data as found in (Chiu, 1994)
	double data[][6] = {{0.038, 0.032, 0.019, 11.429, 16.439, 8.460},
						{0.148, 0.062, 0.051,  9.375,  3.176, 2.250},
						{0.244, 0.139, 0.068,  6.673,  0.193, 0.317},
						{0.222, 0.204, 0.064,  7.738,  3.450, 1.907},
						{0.023, 0.013, 0.039, 10.125,  7.064, 4.746},
						{0.411, 0.192, 0.093,  7.824,  0.179, 0.375},
						{0.132, 0.040, 0.017,  5.378,  0.627, 0.540},
						{0.044, 0.024, 0.021,  9.340,  0.850, 0.768},
						{0.066, 0.011, 0.004, 11.705,  0.859, 0.605},
						{0.537, 0.250, 0.203, 17.406,  2.568, 2.298},
						{1.465, 0.471, 0.436, 17.340,  1.341, 1.708},
						{1.357, 0.433, 0.303,  8.260,  0.411, 0.843},
						{4.071, 1.348, 1.339, 11.590,  1.486, 2.801},
						{2.277, 0.789, 0.569, 18.451,  0.602, 1.239},
						{3.837, 1.355, 1.347, 14.125,  1.077, 2.756},
						{1.590, 0.762, 0.747, 11.703,  0.357, 1.304},
						{3.116, 1.193, 1.487, 19.733,  0.603, 2.385},
						{1.368, 0.517, 0.708, 25.681,  1.265, 1.883},
						{0.937, 0.448, 0.639, 21.691,  0.585, 1.242},
						{3.419, 2.333, 1.766, 13.403,  0.870, 2.946},
						{1.877, 0.763, 0.917, 18.750,  1.565, 2.183},
						{1.872, 0.831, 1.065, 27.661,  0.667, 1.868},
						{3.107, 1.369, 1.561, 16.843,  1.146, 2.772},
						{1.535, 0.605, 0.784, 16.544,  0.399, 1.313},
						{2.326, 0.931, 0.612, 11.454,  0.573, 1.472},
						{3.340, 0.925, 0.775, 11.560,  0.942, 1.852},
						{2.223, 0.850, 0.970, 16.400,  2.524, 3.177},
						{3.274, 1.065, 1.003, 13.193,  0.750, 2.037},
						{1.490, 0.592, 0.686, 13.796,  1.797, 2.037},
						{3.447, 1.366, 1.396, 16.461,  0.705, 2.349},
						{3.035, 1.176, 1.494, 18.963,  0.447, 2.169},
						{0.003, 0.004, 0.002,  6.036,  1.687, 1.272},
						{1.600, 0.506, 0.260,  5.369,  1.385, 1.467},
						{0.398, 0.151, 0.132,  8.159,  0.625, 0.648},
						{2.396, 0.901, 0.542,  8.603,  1.270, 1.838},
						{2.698, 1.268, 0.541,  6.663,  0.758, 1.548},
						{0.414, 0.144, 0.080, 13.119,  0.245, 0.387},
						{0.037, 0.015, 0.005,  9.659,  1.277, 0.966},
						{2.760, 1.333, 1.200, 13.498,  1.064, 2.996},
						{4.476, 1.656, 2.719, 26.266,  0.817, 3.631},
						{1.507, 0.657, 0.706, 15.735,  0.636, 1.446},
						{1.179, 0.434, 0.744, 32.328,  1.353, 2.490},
						{1.727, 0.690, 1.232, 30.573,  1.397, 3.253},
						{3.118, 1.175, 1.995, 30.063,  0.568, 3.400},
						{3.103, 1.147, 2.198, 28.859,  2.115, 5.584},
						{1.057, 0.535, 0.639, 16.101,  0.371, 1.521},
						{1.250, 0.490, 0.731, 21.040,  0.449, 1.169},
						{0.759, 0.368, 0.440, 13.719,  2.780, 3.043},
						{1.219, 0.331, 0.630, 30.581,  0.087, 0.929},
						{3.918, 1.414, 2.196, 21.240,  0.389, 3.059},
						{2.413, 1.004, 1.407, 20.289,  0.675, 2.309},
						{2.487, 0.783, 1.514, 24.101,  1.399, 3.194},
						{1.089, 0.430, 0.734, 18.971,  0.247, 1.114},
						{0.293, 0.117, 0.188, 27.220,  0.108, 0.518},
						{2.101, 0.836, 1.318, 18.686,  0.322, 2.105},
						{1.028, 0.453, 0.657, 15.944,  1.812, 4.431},
						{3.562, 1.360, 1.844, 20.053,  2.000, 5.049},
						{4.247, 1.651, 2.526, 20.093,  0.265, 3.138},
						{2.674, 1.022, 1.649, 26.034,  0.109, 2.015},
						{1.381, 0.515, 0.961, 31.250,  3.363, 2.302},
						{1.131, 0.494, 0.731, 18.873,  1.286, 1.714},
						{2.210, 0.805, 1.460, 25.000,  0.477, 2.383},
						{0.450, 0.188, 0.212, 13.251,  0.418, 0.951},
						{3.206, 1.075, 1.567, 17.099,  0.710, 2.373},
						{3.071, 1.035, 1.562, 20.454,  0.599, 2.670},
						{2.358, 0.732, 1.235, 24.159,  0.592, 1.749},
						{2.507, 0.857, 1.495, 21.843,  0.073, 1.723},
						{6.577, 2.122, 3.710, 26.345,  0.536, 4.902},
						{3.047, 1.175, 2.014, 25.889,  0.684, 3.151},
						{3.551, 1.674, 2.131, 19.401,  1.584, 4.698},
						{0.229, 0.098, 0.146, 30.428,  1.927, 5.127},
						{2.731, 0.811, 1.545, 34.528,  2.473, 6.575},
						{3.510, 1.098, 2.273, 41.679,  0.235, 2.864},
						{3.390, 1.073, 2.329, 48.440,  0.316, 3.074},
						{2.246, 0.809, 1.477, 31.303,  0.111, 1.803}};

	const std::size_t nr = sizeof(data)/sizeof(data[0]);
	const std::size_t nc = sizeof(data[0])/sizeof(data[0][0]);

	std::vector< std::vector<fl::scalar> > dataset(nr);
	for (std::size_t i = 0; i < nr; ++i)
	{
		dataset[i].assign(data[i], data[i]+nc);
	}

	subclust.setRadii(0.5, 6);
	subclust.setAcceptanceRatio(0.5);
	subclust.setRejectionRatio(0.15);
	subclust.setSquashingFactor(1.25);
	subclust.cluster(dataset);
}

template <typename T>
bool CheckEqual(const T& v1, const T& v2, double tol = DefaultTolerance)
{
	if (!fl::detail::FloatTraits<T>::EssentiallyEqual(v1, v2, tol))
	{
		return false;
	}

	return true;
}

template <typename T>
bool CheckEqual(const std::vector<T>& v1, const std::vector<T>& v2, double tol = DefaultTolerance)
{
	if (v1.size() != v2.size())
	{
		return false;
	}

	for (std::size_t i = 0,
					 n = v1.size();
		 i < n;
		 ++i)
	{
		if (!CheckEqual(v1[i], v2[i], tol))
		{
			return false;
		}
	}

	return true;
}

template <typename T>
bool CheckEqual(const std::vector< std::vector<T> >& A1, const std::vector< std::vector<T> >& A2, double tol = DefaultTolerance)
{
	if (A1.size() != A2.size())
	{
		return false;
	}

	for (std::size_t i = 0,
					 nr = A1.size();
		 i < nr;
		 ++i)
	{
		if (A1[i].size() != A2[i].size())
		{
			return false;
		}

		for (std::size_t j = 0,
						 nc = A1[i].size();
			 j < nc;
			 ++j)
		{
			if (!CheckEqual(A1[i][j], A2[i][j], tol))
			{
				return false;
			}
		}
	}

	return true;
}

bool CheckEqual(const fl::cluster::SubtractiveClustering& subclust1, const fl::cluster::SubtractiveClustering& subclust2, double tol = DefaultTolerance)
{
	if (!CheckEqual(subclust1.radii(), subclust2.radii(), tol)
		|| !CheckEqual(subclust1.getSquashingFactor(), subclust2.getSquashingFactor(), tol)
		|| !CheckEqual(subclust1.getAcceptanceRatio(), subclust2.getAcceptanceRatio(), tol)
		|| !CheckEqual(subclust1.getRejectionRatio(), subclust2.getRejectionRatio(), tol)
		|| !CheckEqual(subclust1.lowerBounds(), subclust2.lowerBounds(), tol)
		|| !CheckEqual(subclust1.upperBounds(), subclust2.upperBounds(), tol)
		|| !CheckEqual(subclust1.centers(), subclust2.centers(), tol)
		|| subclust1.numOfClusters() != subclust2.numOfClusters()
		|| !CheckEqual(subclust1.rangeOfInfluence(), subclust2.rangeOfInfluence(), tol))
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
		fl::cluster::SubtractiveClustering cluster;

		if (cluster.radii().size() != 0
			|| cluster.lowerBounds().size() != 0
			|| cluster.upperBounds().size() != 0
			|| cluster.centers().size() != 0
			|| cluster.numOfClusters() != 0
			|| cluster.rangeOfInfluence().size() != 0)
		{
			throw std::runtime_error("Failed construction test: default construction");
		}
	}
}

/// Test copy construction & similar functionalities
void TestCopy()
{
	// Copy ctor (from ANFIS engine)
	{
		fl::cluster::SubtractiveClustering subclust1;

		detail::Setup(subclust1);

		fl::cluster::SubtractiveClustering subclust2(subclust1);

		if (!detail::CheckEqual(subclust1, subclust2))
		{
			throw std::runtime_error("Failed copy test: copy-construction");
		}
	}

	// Assignment operator
	{
		fl::cluster::SubtractiveClustering subclust1;

		detail::Setup(subclust1);

		fl::cluster::SubtractiveClustering subclust2;

		subclust2 = subclust1;

		if (!detail::CheckEqual(subclust1, subclust2))
		{
			throw std::runtime_error("Failed copy test: copy-assignment");
		}
	}
}

/// Test functional behavior 
void TestFunctional()
{
	std::vector< std::vector<fl::scalar> > centers(3);
	centers[0].resize(6);
	centers[0][0] = 1.8770; centers[0][1] = 0.7630; centers[0][2] = 0.9170; centers[0][3] = 18.7500; centers[0][4] = 1.5650; centers[0][5] = 2.1830;
	centers[1].resize(6);                                                                                                                           
	centers[1][0] = 0.3980; centers[1][1] = 0.1510; centers[1][2] = 0.1320; centers[1][3] =  8.1590; centers[1][4] = 0.6250; centers[1][5] = 0.6480;
	centers[2].resize(6);                                                                                                                           
	centers[2][0] = 3.1160; centers[2][1] = 1.1930; centers[2][2] = 1.4870; centers[2][3] = 19.7330; centers[2][4] = 0.6030; centers[2][5] = 2.3850;

	std::vector<fl::scalar> sigmas(6);
    sigmas[0] = 1.1621; sigmas[1] = 0.4117; sigmas[2] = 0.6555; sigmas[3] = 7.6139; sigmas[4] = 2.8931; sigmas[5] = 1.4395;


	fl::cluster::SubtractiveClustering subclust;

	detail::Setup(subclust);

	if (subclust.numOfClusters() != 3)
	{
		std::cerr << "# Clusters: " << subclust.numOfClusters() << std::endl;
		throw std::runtime_error("Failed functional test: number of clusters");
	}
	if (!detail::CheckEqual(subclust.centers(), centers, 1e-4))
	{
		throw std::runtime_error("Failed functional test: cluster centers");
	}
	if (!detail::CheckEqual(subclust.rangeOfInfluence(), sigmas, 1e-4))
	{
		throw std::runtime_error("Failed functional test: range of influences");
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
