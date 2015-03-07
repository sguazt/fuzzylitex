#ifndef FL_ANN_OPERATIONS_H
#define FL_ANN_OPERATIONS_H

namespace fl { namespace ann {

template <typename NNetT, typename AlgT, typename DT>
void TrainSingleEpoch(NNetT& nnet, AlgT& alg, const DT& ds)
{
	alg.trainSingleEpoch(ds);
}

}} // Namespace fl::ann

#endif // FL_ANN_OPERATIONS_H
