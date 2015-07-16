/**
 * \file fl/cluster/subtractive.h
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

#ifndef FL_CLUSTER_SUBTRACTIVE_H
#define FL_CLUSTER_SUBTRACTIVE_H


#include <cmath>
#include <cstddef>
#include <fl/detail/math.h>
#include <fl/detail/traits.h>
#include <fl/fuzzylite.h>
#include <fl/macro.h>
#include <vector>


namespace fl { namespace cluster {

/**
 * \brief Estimates the cluster centers in a set of data by means of the
 *  subtractive clustering method (Chiu, 1994).
 *
 * Subtractive clustering (Chiu, 1994) is a fast one-pass algorithm for
 * estimating the number of clusters and the cluster centers in a set of data.
 * It is an extension of the mountain clustering method proposed by
 * (Yager et al., 1994).
 *
 * This algorithm assumes each data point is a potential cluster center and
 * computes a measure of the likelihood that each data point would define the
 * cluster center, based on the density surrounding data points.
 * Specifically:
 * - Normalizes data points to make them fall within a unit hyperbox.
 *   Normalization is performed according to the bounds given as input to
 *   the algorithm or, alternatively, according to the min and max values
 *   found in the input data points.
 * - Considers each data point as a potential cluster center and defines a
 *   measure of the potential of data point \f$x_i\f$ as:
 *   \f[
 *    P_i = \sum_{j=1}^n e^{-\|\alpha \circ (x_i-x_j)\|^2}
 *   \f]
 *   where \f$n\f$ is the number of data points, \f$\alpha\f$ is given by:
 *   \f[
 *    \alpha = [\frac{2}{r_1^2},\ldots,\frac{2}{r_m^2}]^T
 *   \f]
 *   \f$r_k\f$, for \f$k=1,\ldots,m\f$, is the radius defining the neighborhood
 *   for the \f$k\f$th data dimension, and \f$\circ\f$ denotes the element-wise
 *   product.
 * - Selects the data point with highest potential to be the first cluster
 *   center.
 * - Removes all data points in the neighborhood of the first cluster center,
 *   in order to determine the next data cluster and its center location.
 * - Iterates on this process until all of the data is withing a cluster
 *   center's range of influence (controlled by the \c radii parameter).
 * .
 *
 * The algorithm takes the following parameters:
 * - <em>cluster radii</em>: a vector of entries between 0 and 1 that specifies
 *   a cluster center's range of influence in each of the data dimensions
 *   (assuming the data falls within a unit hyperbox).
 *   Smaller cluster radius will usually yield more (smaller) clusters in the
 *   data.
 * - <em>acceptance ratio</em>: a positive threshold, as a fraction of the
 *   potential of the first cluster center, above which another data point will
 *   be accepted as a cluster center.
 *   If not provided, 0.5 is used as default value.
 * - <em>rejection ratio</em>: a positive threshold, as a fraction of the
 *   potential of the first cluster center, below which another data point will
 *   be rejected as a cluster center.
 *   If not provided, 0.15 is used as default value.
 * - <em>squash factor</em>: a multiplier for the range of influence (radius)
 *   that is used to determine the neighborhood of a cluster center within which
 *   the existence of other cluster centers are discouraged.
 *   If not provided, 1.25 is used as default value.
 * - <em>bounds</em>: a vector of lower and upper bounds for each data dimension
 *   used to normalized the data points into e unit hyperbox.
 *   If not provided, the bounds are directly obtained from the input data points
 *   by taking the min and max values.
 * .
 *
 * The cluster estimates are often used to initialize other iterative
 * optimization-based clustering methods (e.g., fuzzy c-means) and model
 * identification methods (e.g, ANFIS).
 *
 * Rerefences:
 * -# S.L. Chiu, "Fuzzy Model Identification Based on Cluster Estimation," Journal of Intelligent and Fuzzy Systems, 2(3):267-278, 1994
 * -# R. Yager and D. Filev, "Generation of Fuzzy Rules by Mountain Clustering," Journal of Intelligent & Fuzzy Systems, 2(3):209-219, 1994.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
class FL_API SubtractiveClustering
{
private:
	static const fl::scalar DefaultRadius;
	static const fl::scalar DefaultSquashingFactor;
	static const fl::scalar DefaultAcceptanceRatio;
	static const fl::scalar DefaultRejectionRatio;

public:
	/// The default constructor
	SubtractiveClustering()
	: squashFactor_(DefaultSquashingFactor),
	  acceptRatio_(DefaultAcceptanceRatio),
	  rejectRatio_(DefaultRejectionRatio)
	{
	}

	/// The copy constructor
	SubtractiveClustering(const SubtractiveClustering& other)
	: radii_(other.radii_),
	  squashFactor_(other.squashFactor_),
	  acceptRatio_(other.acceptRatio_),
	  rejectRatio_(other.rejectRatio_),
	  lbounds_(other.lbounds_),
	  ubounds_(other.lbounds_),
	  centers_(other.centers_),
	  sigma_(other.sigma_)
	{
	}

	FL_DEFAULT_MOVE(SubtractiveClustering)

	SubtractiveClustering& operator=(const SubtractiveClustering& rhs)
	{
		if (this != &rhs)
		{
			radii_ = rhs.radii_;
			squashFactor_ = rhs.squashFactor_;
			acceptRatio_ = rhs.acceptRatio_;
			rejectRatio_ = rhs.rejectRatio_;
			lbounds_ = rhs.lbounds_;
			ubounds_ = rhs.lbounds_;
			centers_ = rhs.centers_;
			sigma_ = rhs.sigma_;
		}

		return *this;
	}

	/**
	 * Sets the cluster radius for each data dimension used to determine the
	 * range of influence of a (dimension of a) cluster center.
	 */
	void setRadii(const std::vector<fl::scalar>& radii)
	{
		radii_ =  radii;
	}

	/**
	 * Sets the cluster radius for each data dimension used to determine the
	 * range of influence of a (dimension of a) cluster center.
	 */
	void setRadii(fl::scalar radii, std::size_t n)
	{
		radii_ =  std::vector<fl::scalar>(n, radii);
	}

	/**
	 * Returns the cluster radius for each data dimension used to determine the
	 * range of influence of a (dimension of a) cluster center.
	 */
	std::vector<fl::scalar> radii() const
	{
		return radii_;
	}

	/**
	 * Sets the squashing factor used to discourage the formation of new cluster
	 * centers within an existing cluster.
	 */
	void setSquashingFactor(fl::scalar value)
	{
		squashFactor_ = value;
	}

	/**
	 * Returns the squashing factor used to discourage the formation of new
	 * cluster centers within an existing cluster.
	 */
	fl::scalar getSquashingFactor() const
	{
		return squashFactor_;
	}

	/**
	 * Sets the acceptance ratio threshold above which a data point is accepted
	 * as a cluster center.
	 */
	void setAcceptanceRatio(fl::scalar value)
	{
		acceptRatio_ = value;
	}

	/**
	 * Returns the acceptance ratio threshold above which a data point is
	 * accepted as a cluster center.
	 */
	fl::scalar getAcceptanceRatio() const
	{
		return acceptRatio_;
	}

	/**
	 * Sets the rejection ratio threshold below which a data point is rejected
	 * as a cluster center.
	 */
	void setRejectionRatio(fl::scalar value)
	{
		rejectRatio_ = value;
	}

	/**
	 * Returns the rejection ratio threshold below which a data point is
	 * rejected as a cluster center.
	 */
	fl::scalar getRejectionRatio() const
	{
		return rejectRatio_;
	}

	/**
	 * Sets the lower bound for each data dimension used to normalize data
	 * points into a unit hyperbox
	 */
	void setLowerBounds(const std::vector<fl::scalar>& value)
	{
		lbounds_ = value;
	}

	/**
	 * Returns the lower bound for each data dimension used to normalize data
	 * points into a unit hyperbox
	 */
	std::vector<fl::scalar> lowerBounds() const
	{
		return lbounds_;
	}

	/**
	 * Sets the upper bound for each data dimension used to normalize data
	 * points into a unit hyperbox
	 */
	void setUpperBounds(const std::vector<fl::scalar>& value)
	{
		ubounds_ = value;
	}

	/**
	 * Returns the upper bound for each data dimension used to normalize data
	 * points into a unit hyperbox
	 */
	std::vector<fl::scalar> upperBounds() const
	{
		return ubounds_;
	}

	/**
	 * Sets the lower and upper bounds for each data dimension used to normalize
	 * data points into a unit hyperbox
	 */
	void setBounds(const std::vector<fl::scalar>& lower, const std::vector<fl::scalar>& upper)
	{
		this->setLowerBounds(lower);
		this->setUpperBounds(upper);
	}

	/**
	 * Returns the lower and upper bounds for each data dimension used to
	 * normalize data points into a unit hyperbox
	 */
	std::vector<std::vector<fl::scalar> > bounds() const
	{
		std::vector<std::vector<fl::scalar> > res;
		res.push_back(lbounds_);
		res.push_back(ubounds_);

		return res;
	}

	void reset()
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

	template <typename MatrixT>
	void cluster(const MatrixT& data)
	{
		const std::size_t nr = data.size(); // Number of data points
		if (nr == 0)
		{
			FL_THROW("Data set must have at least one point");
		}

		const std::size_t nc = data[0].size(); // Number of data parameters

		// Adjust radii parameters (if needed)
		if (radii_.size() < nc)
		{
			// Enlarge and set default value
			radii_.insert(radii_.end(), nc-radii_.size(), DefaultRadius);
		}
		else if (radii_.size() > nc)
		{
			// Shrink to fit current input data dimension
			radii_.resize(nc);
		}

		// Precomputes distance multipliers for accumulating and squashing potentials
		std::vector<fl::scalar> squash(radii_.size());
		std::vector<fl::scalar> accum(radii_.size());
		for (std::size_t i = 0; i < nc; ++i)
		{
			accum[i] = 1.0/radii_[i];
			squash[i] = 1.0/(squashFactor_*radii_[i]);
		}

		// Normalize data

		// - Checks/sets data bounds
		bool clearLBounds = false;
		bool clearUBounds = false;
		if (lbounds_.size() == 0 || ubounds_.size() == 0)
		{
			// No bound given => Use min and max values
			if (lbounds_.size() == 0)
			{
				clearLBounds = true;
				lbounds_.resize(nc, std::numeric_limits<fl::scalar>::max());
			}
			if (ubounds_.size() == 0)
			{
				clearUBounds = true;
				ubounds_.resize(nc, std::numeric_limits<fl::scalar>::min());
			}
			if (clearLBounds || clearUBounds)
			{
				for (std::size_t j = 0; j < nc; ++j)
				{
					for (std::size_t i = 0; i < nr; ++i)
					{
						if (clearLBounds && lbounds_[j] > data[i][j])
						{
							lbounds_[j] = data[i][j];
						}
						if (clearUBounds && ubounds_[j] < data[i][j])
						{
							ubounds_[j] = data[i][j];
						}
					}
				}
			}
			// For zero-range data, compute a small artificial range
			for (std::size_t j = 0; j < nc; ++j)
			{
				if (fl::detail::FloatTraits<fl::scalar>::ApproximatelyEqual(lbounds_[j], ubounds_[j]))
				{
					if (clearLBounds)
					{
						lbounds_[j] -= 0.0001*(1+std::abs(lbounds_[j]));
					}
					if (clearUBounds)
					{
						ubounds_[j] += 0.0001*(1+std::abs(ubounds_[j]));
					}
				}
			}
		}
		else
		{
			// Check size
			if (lbounds_.size() != nc || ubounds_.size() != nc)
			{
				FL_THROW("Wrong dimension for data bound vectors");
			}
			// Check zero-range data
			for (std::size_t i = 0; i < nc; ++i)
			{
				if (fl::detail::FloatTraits<fl::scalar>::ApproximatelyEqual(lbounds_[i], ubounds_[i]))
				{
					FL_THROW("Found zero data-range in data bound vector");
				}
			}
		}
		// - Does data normalization
		std::vector< std::vector<fl::scalar> > datan(nr);
		for (std::size_t i = 0; i < nr; ++i)
		{
			datan[i].resize(nc);
			for (std::size_t j = 0; j < nc; ++j)
			{
				datan[i][j] = std::min(std::max((data[i][j]-lbounds_[j])/(ubounds_[j]-lbounds_[j]), 0.0), 1.0);
			}
		}

		// Computes potential values

		// - Potential values for each data point
		std::vector<fl::scalar> potentials(nr, 0);

		// - Computes the initial potential values.
		//   Also, find the data point with the highest potential.
		//   The highest potential will be used as a reference for
		//   accepting/rejecting other data points as cluster centers.
		fl::scalar maxPotential = -1; // -1 is OK since potentials cannot be negative values
		std::size_t maxPotentialIdx = 0;
		for (std::size_t i = 0; i < nr; ++i)
		{
			const std::vector<fl::scalar>& point = datan[i];
			for (std::size_t k = 0; k < nr; ++k)
			{
				fl::scalar distSq = 0; // the squared distance
				for (std::size_t j = 0; j < nc; ++j)
				{
					const fl::scalar dist = (point[j]-datan[k][j])*accum[j];
					distSq += fl::detail::Sqr(dist);
				}
				potentials[i] += std::exp(-4.0*distSq);
			}

			// Update max potential
			if (maxPotential < potentials[i])
			{
				maxPotential = potentials[i];
				maxPotentialIdx = i;
			}
		}

		// Start iteratively finding cluster centers and subtracting potential
		// from neighboring data points. 
		bool findMore = true;
		const fl::scalar refPotential = maxPotential;
		centers_.clear();
		while (findMore && maxPotential > 0)
		{
			const fl::scalar maxPotentialRatio = maxPotential/refPotential;
			const std::vector<fl::scalar>& maxPotentialPoint = datan[maxPotentialIdx];

			bool removePoint = false;
			findMore = false;
			if (maxPotentialRatio > acceptRatio_)
			{
				// The new peak value is significant -> accept
				findMore = true;
			}
			else if (maxPotentialRatio > rejectRatio_)
			{
				// Accept this data point only if it achieves a good balance between having a reasonable potential and being far from all existing cluster centers
				fl::scalar minDistSq = -1;//std::numeric_limits<fl::scalar>::infinity();

				for (std::size_t i = 0,
								 ni = centers_.size();
					 i < ni;
					 ++i)
				{
					fl::scalar distSq = 0;
					for (std::size_t j = 0; j < nc; ++j)
					{
						const fl::scalar dist = (maxPotentialPoint[j]-centers_[i][j])*accum[j];
						distSq += fl::detail::Sqr(dist);
					}
					if (minDistSq < 0 || distSq < minDistSq)
					{
						minDistSq = distSq;
					}
				}

				fl::scalar minDist = std::sqrt(minDistSq);
				if ((maxPotentialRatio+minDist) >= 1)
				{
					// Tentatively accept this data point as a cluster center
					findMore = true;
				}
				else
				{
					// Remove this data point from further consideration and continue
					findMore = true;
					removePoint = true;
				}
			}

			if (findMore)
			{
				if (removePoint)
				{
					potentials[maxPotentialIdx] = 0;
					// Recompute max potential
					maxPotential = -1; // -1 is OK since potentials cannot be negative values
					for (std::size_t i = 0; i < nr; ++i)
					{
						if (maxPotential < potentials[i])
						{
							maxPotential = potentials[i];
							maxPotentialIdx = i;
						}
					}
				}
				else
				{
					// Adds the data point to the list of cluster centers
					centers_.push_back(maxPotentialPoint);

					FL_DEBUG_TRACE("Found cluster #" << centers_.size() << ", potential = " << maxPotentialRatio);

					// subtract potential from data points near the new cluster center
					for (std::size_t i = 0; i < nr; ++i)
					{
						fl::scalar distSq = 0;
						for (std::size_t j = 0; j < nc; ++j)
						{
							const fl::scalar dist = (maxPotentialPoint[j]-datan[i][j])*squash[j];
							distSq += fl::detail::Sqr(dist);
						}
						potentials[i] = std::max(potentials[i]-maxPotential*std::exp(-4.0*distSq), 0.0);
					}

					// Finds the data point with the highest remaining potential
					maxPotential = -1; // OK since potentials cannot be negative values
					for (std::size_t i = 0; i < nr; ++i)
					{
						if (maxPotential < potentials[i])
						{
							maxPotential = potentials[i];
							maxPotentialIdx = i;
						}
					}
				}
			}
		}

		// Scale the cluster centers from the normalized values back to values in
		// the original range
		for (std::size_t i = 0,
						 numClusters = centers_.size();
			 i < numClusters;
			 ++i)
		{
			for (std::size_t j = 0; j < nc; ++j)
			{
				centers_[i][j] = centers_[i][j]*(ubounds_[j]-lbounds_[j]) + lbounds_[j];
			}
		}

		// Compute the range of influence of the cluster centers for each data
		// dimension
		sigma_.resize(nc);
		for (std::size_t i = 0; i < nc; ++i)
		{
			sigma_[i] = (radii_[i]*(ubounds_[i]-lbounds_[i]))/std::sqrt(8.0);
		}

		//if (clearLBounds)
		//{
		//	lbounds_.clear();
		//}
		//if (clearUBounds)
		//{
		//	ubounds_.clear();
		//}
	}

	/// Returns the found cluster centers
	std::vector< std::vector<fl::scalar> > clusterCenters() const
	{
		return centers_;
	}

	/// Returns the number of found clusters
	std::size_t numOfClusters() const
	{
		return centers_.size();
	}

	/**
	 * Returns the range of influence of the found clusters in each of the data
	 * dimensions.
	 *
	 * All cluster centers share the same set of sigma values.
	 */
	std::vector<fl::scalar> rangeOfInfluence() const
	{
		return sigma_;
	}


private:
	std::vector<fl::scalar> radii_; ///< The vector of cluster sizes in each of the data dimensions
	fl::scalar squashFactor_; ///< Used to multiply the cluster sizes to determine the neighborhood of a cluster center within which the existence of other cluster centers are discouraged
	fl::scalar acceptRatio_; ///< The threshold, as a fraction of the potential of the first cluster center, above which another data point will be accepted as a cluster center
	fl::scalar rejectRatio_; ///< The threshold, as a fraction of the potential of the first cluster center, below which another data point will be rejected as a cluster center
	std::vector<fl::scalar> lbounds_; ///< Lower bounds (one for each data dimension) used to normalized data point within a unit hyperbox
	std::vector<fl::scalar> ubounds_; ///< Upper bounds (one for each data dimension) used to normalized data point within a unit hyperbox
	std::vector< std::vector<fl::scalar> > centers_; ///< The found cluster centers
	std::vector<fl::scalar> sigma_; ///< Range of influence of the cluster centers
}; // SubtractiveClustering

const fl::scalar SubtractiveClustering::DefaultRadius = 0.5;

const fl::scalar SubtractiveClustering::DefaultSquashingFactor = 1.25;

const fl::scalar SubtractiveClustering::DefaultAcceptanceRatio = 0.5;

const fl::scalar SubtractiveClustering::DefaultRejectionRatio = 0.15;

}} // Namespace fl::cluster


#endif // FL_CLUSTER_SUBTRACTIVE_H
