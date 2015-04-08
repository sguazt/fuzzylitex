/**
 * \file fl/dataset.tpp
 *
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly.
 */

#ifndef FL_DATASET_TPP
#define FL_DATASET_TPP


#include <cstddef>
#include <fl/commons.h>
#include <fl/fuzzylite.h>
#include <stdexcept>
#include <vector>


namespace fl {

template <typename ValueT>
DataSetEntry<ValueT>::DataSetEntry()
{
}

template <typename ValueT>
template <typename InIterT, typename OutIterT>
DataSetEntry<ValueT>::DataSetEntry(InIterT inFirst, InIterT inLast,
								   OutIterT outFirst, OutIterT outLast)
: in_(inFirst, inLast),
  out_(outFirst, outLast)
{
}

template <typename ValueT>
template <typename IterT>
void DataSetEntry<ValueT>::setInputs(IterT first, IterT last)
{
	in_.clear();
	in_.assign(first, last);
}

template <typename ValueT>
template <typename IterT>
void DataSetEntry<ValueT>::setOutputs(IterT first, IterT last)
{
	out_.clear();
	out_.assign(first, last);
}

template <typename ValueT>
std::size_t DataSetEntry<ValueT>::numOfInputs() const
{
	return in_.size();
}

template <typename ValueT>
std::size_t DataSetEntry<ValueT>::numOfOutputs() const
{
	return out_.size();
}

template <typename ValueT>
ValueT DataSetEntry<ValueT>::getField(std::size_t index) const
{
	if (index >= (in_.size()+out_.size()))
	{
		FL_THROW2(std::invalid_argument, "Index is out-of-range");
	}

	return index < in_.size() ? in_.at(index)
							  : out_.at(index-in_.size());
}


///////////
// DataSet
///////////


template <typename ValueT>
DataSet<ValueT>::DataSet(std::size_t ni, std::size_t no)
: ni_(ni),
  no_(no)
{
}

template <typename ValueT>
void DataSet<ValueT>::add(const DataSetEntry<ValueT>& entry)
{
	if (entry.numOfInputs() != ni_)
	{
		FL_THROW2(std::invalid_argument, "Unexpected number of inputs in the entry");
	}
	if (entry.numOfOutputs() != no_)
	{
		FL_THROW2(std::invalid_argument, "Unexpected number of outputs in the entry");
	}

	entries_.push_back(entry);
}

template <typename ValueT>
typename DataSet<ValueT>::EntryIterator DataSet<ValueT>::insert(typename DataSet<ValueT>::EntryIterator pos,
																const DataSetEntry<ValueT>& entry)
{
	if (entry.numOfInputs() != ni_)
	{
		FL_THROW2(std::invalid_argument, "Unexpected number of inputs in the entry");
	}
	if (entry.numOfOutputs() != no_)
	{
		FL_THROW2(std::invalid_argument, "Unexpected number of outputs in the entry");
	}

	return entries_.insert(pos, entry);
}

template <typename ValueT>
typename DataSet<ValueT>::EntryIterator DataSet<ValueT>::insert(typename DataSet<ValueT>::ConstEntryIterator pos,
																const DataSetEntry<ValueT>& entry)
{
	if (entry.numOfInputs() != ni_)
	{
		FL_THROW2(std::invalid_argument, "Unexpected number of inputs in the entry");
	}
	if (entry.numOfOutputs() != no_)
	{
		FL_THROW2(std::invalid_argument, "Unexpected number of outputs in the entry");
	}

	return entries_.insert(pos, entry);
}

//template <typename ValueT>
//typename DataSet<ValueT>::EntryIterator DataSet<ValueT>::insert(typename DataSet<ValueT>::ConstEntryIterator pos,
//																  const DataSetEntry<ValueT>& entry)
//{
//	if (entry.numOfInputs() != ni_)
//	{
//		FL_THROW2(std::invalid_argument, "Unexpected number of inputs in the entry");
//	}
//	if (entry.numOfOutputs() != no_)
//	{
//		FL_THROW2(std::invalid_argument, "Unexpected number of outputs in the entry");
//	}
//
//	return entries_.insert(pos, entry);
//}

template <typename ValueT>
typename DataSet<ValueT>::EntryIterator DataSet<ValueT>::erase(typename DataSet<ValueT>::EntryIterator pos)
{
	return entries_.erase(pos);
}

template <typename ValueT>
typename DataSet<ValueT>::EntryIterator DataSet<ValueT>::erase(typename DataSet<ValueT>::ConstEntryIterator pos)
{
	return entries_.erase(pos);
}

template <typename ValueT>
void DataSet<ValueT>::set(const DataSetEntry<ValueT>& entry, std::size_t idx) const
{
	return entries_[idx] = entry;
}

template <typename ValueT>
const DataSetEntry<ValueT>& DataSet<ValueT>::get(std::size_t idx) const
{
	return entries_.at(idx);
}

template <typename ValueT>
std::size_t DataSet<ValueT>::size() const
{
	return entries_.size();
}

template <typename ValueT>
void DataSet<ValueT>::setNumOfInputs(std::size_t val)
{
	ni_ = val;
}

template <typename ValueT>
std::size_t DataSet<ValueT>::numOfInputs() const
{
	return ni_;
}

template <typename ValueT>
void DataSet<ValueT>::setNumOfOutputs(std::size_t val)
{
	no_ = val;
}

template <typename ValueT>
std::size_t DataSet<ValueT>::numOfOutputs() const
{
	return no_;
}

} // Namespace fl

#endif // FL_DATASET_TPP
