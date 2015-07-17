/**
 * \file cluster/subtractive.cpp
 *
 * \brief Subtractive clustering (Chiu, 1994)
 *
 * Rerefences:
 * -# S.L. Chiu, "Fuzzy Model Identification Based on Cluster Estimation," Journal of Intelligent and Fuzzy Systems, 2(3):267-278, 1994
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
#include <fl/cluster/subtractive.h>
#include <fl/fuzzylite.h>
#include <fl/macro.h>
#include <stdexcept>
#include <vector>


namespace fl { namespace cluster {

const fl::scalar SubtractiveClustering::DefaultRadius = 0.5;

const fl::scalar SubtractiveClustering::DefaultSquashingFactor = 1.25;

const fl::scalar SubtractiveClustering::DefaultAcceptanceRatio = 0.5;

const fl::scalar SubtractiveClustering::DefaultRejectionRatio = 0.15;

SubtractiveClustering::SubtractiveClustering()
: squashFactor_(DefaultSquashingFactor),
  acceptRatio_(DefaultAcceptanceRatio),
  rejectRatio_(DefaultRejectionRatio)
{
}

SubtractiveClustering::SubtractiveClustering(const SubtractiveClustering& other)
: radii_(other.radii_),
  squashFactor_(other.squashFactor_),
  acceptRatio_(other.acceptRatio_),
  rejectRatio_(other.rejectRatio_),
  lbounds_(other.lbounds_),
  ubounds_(other.ubounds_),
  centers_(other.centers_),
  sigma_(other.sigma_)
{
}

SubtractiveClustering& SubtractiveClustering::operator=(const SubtractiveClustering& rhs)
{
	if (this != &rhs)
	{
		radii_ = rhs.radii_;
		squashFactor_ = rhs.squashFactor_;
		acceptRatio_ = rhs.acceptRatio_;
		rejectRatio_ = rhs.rejectRatio_;
		lbounds_ = rhs.lbounds_;
		ubounds_ = rhs.ubounds_;
		centers_ = rhs.centers_;
		sigma_ = rhs.sigma_;
	}

	return *this;
}

void SubtractiveClustering::setRadii(const std::vector<fl::scalar>& radii)
{
	radii_ =  radii;
}

void SubtractiveClustering::setRadii(fl::scalar radii, std::size_t n)
{
	radii_ =  std::vector<fl::scalar>(n, radii);
}

std::vector<fl::scalar> SubtractiveClustering::radii() const
{
	return radii_;
}

void SubtractiveClustering::setSquashingFactor(fl::scalar value)
{
	squashFactor_ = value;
}

fl::scalar SubtractiveClustering::getSquashingFactor() const
{
	return squashFactor_;
}

void SubtractiveClustering::setAcceptanceRatio(fl::scalar value)
{
	acceptRatio_ = value;
}

fl::scalar SubtractiveClustering::getAcceptanceRatio() const
{
	return acceptRatio_;
}

void SubtractiveClustering::setRejectionRatio(fl::scalar value)
{
	rejectRatio_ = value;
}

fl::scalar SubtractiveClustering::getRejectionRatio() const
{
	return rejectRatio_;
}

void SubtractiveClustering::setLowerBounds(const std::vector<fl::scalar>& value)
{
	lbounds_ = value;
}

std::vector<fl::scalar> SubtractiveClustering::lowerBounds() const
{
	return lbounds_;
}

void SubtractiveClustering::setUpperBounds(const std::vector<fl::scalar>& value)
{
	ubounds_ = value;
}

std::vector<fl::scalar> SubtractiveClustering::upperBounds() const
{
	return ubounds_;
}

void SubtractiveClustering::setBounds(const std::vector<fl::scalar>& lower, const std::vector<fl::scalar>& upper)
{
	if (lower.size() != upper.size())
	{
		FL_THROW2(std::invalid_argument, "Vector dimensions mismatch");
	}

	this->setLowerBounds(lower);
	this->setUpperBounds(upper);
}

std::vector<std::vector<fl::scalar> > SubtractiveClustering::bounds() const
{
	std::vector<std::vector<fl::scalar> > res(lbounds_.size());
	for (std::size_t i = 0,
					 n = res.size();
		 i < n;
		 ++i)
	{
		res[i].push_back(lbounds_[i]);
		res[i].push_back(ubounds_[i]);
	}

	return res;
}

void SubtractiveClustering::reset()
{
	lbounds_.clear();
	ubounds_.clear();
	radii_.clear();
	squashFactor_ = DefaultSquashingFactor;
	acceptRatio_ = DefaultAcceptanceRatio;
	rejectRatio_ = DefaultRejectionRatio;
	centers_.clear();
	sigma_.clear();
}

std::vector< std::vector<fl::scalar> > SubtractiveClustering::centers() const
{
    return centers_;
}

std::size_t SubtractiveClustering::numOfClusters() const
{
	return centers_.size();
}

std::vector<fl::scalar> SubtractiveClustering::rangeOfInfluence() const
{
	return sigma_;
}

}} // Namespace fl::cluster
