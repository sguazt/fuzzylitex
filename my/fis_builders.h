#ifndef FL_FIS_BUILDERS_H
#define FL_FIS_BUILDERS_H

#include <cstddef>
#include <fl/dataset.h>
#include <fl/fuzzylite.h>
#include <string>
#include <vector>


namespace fl {

/**
 * FIS builder based on the <em>grid partitioning</em> method [1].
 *
 * With the <em>grid partitioning</em> method, the multidimensional input-output
 * data space is divided into fixed partitions.
 *
 * Despite its simplicity, this method comes with several disadvantages:
 * - This method suffers of the <em>curse of dimensionality</em> problem,
 *   whereby the number of generated rules grows exponentially with the number
 *   of input variables, of input terms, of output variables, and of output
 *   terms [1,2]. Specifically, with \f$n\f$ inputs and \f$k\f$ membership
 *   functions for each input, the total number of fuzzy rules generated is
 *   \f$k^n\f$. 
 * - Also, some of the generated rules may turn out to be meaningless [2].
 * .
 *
 * References:
 * -# F.O. Karray et al., "Soft Computing and Intelligent Systems Design: Theory, Tools, and Applications,"
 * -# J.-S. R. Jang et al., "Neuro-Fuzzy and Soft Computing: A Computational Approach to Learning and Machine Intelligence," Prentice-Hall, Inc., 1997.
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename EngineT>
class GridPartitionFisBuilder
{
private:
	static const std::size_t DefaultNumberOfInputTerms;
	static const std::string DefaultInputTerm;
	static const std::size_t DefaultNumberOfOutputTerms;
	static const std::string DefaultOutputTerm;


public:
	GridPartitionFisBuilder();

	GridPartitionFisBuilder(std::size_t numInputTerms,
							const std::string& inputTermClass,
							const std::string& outputTermClass);

	template <typename NumInTermsIterT,
			  typename InTermClassIterT>
	GridPartitionFisBuilder(NumInTermsIterT numInTermsFirst, NumInTermsIterT numInTermsLast,
							InTermClassIterT inTermClassFirst, InTermClassIterT inTermClassLast,
							const std::string& outTermClass);

	FL_unique_ptr<EngineT> build(const fl::DataSet<fl::scalar>& data);


private:
	std::vector<std::size_t> numInTerms_;
	std::vector<std::string> inTerms_;
	std::vector<std::size_t> numOutTerms_;
	std::vector<std::string> outTerms_;
}; // GridPartitionFisBuilder

} // Namespace fl


#include "fl/fis_builders.tpp"


#endif // FL_FIS_BUILDERS_H
