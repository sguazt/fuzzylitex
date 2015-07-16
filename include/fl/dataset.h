/**
 * \file fl/dataset.h
 *
 * \brief Header file for classes representing datasets
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2014 Marco Guazzone (marco.guazzone@gmail.com)
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

#ifndef FL_DATASET_H
#define FL_DATASET_H


#include <cstddef>
#include <fl/macro.h>
#include <fl/fuzzylite.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>


namespace fl {

/**
 * Single entry of a dataset
 *
 * \tparam ValueT Types for values stored in the dataset entry
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT = fl::scalar>
class DataSetEntry
{
private:
    typedef std::vector<ValueT> Container;

    FL_MAKE_ITERATORS(public, Container, Input, input, in_) 
    FL_MAKE_ITERATORS(public, Container, Output, output, out_) 


public:
    DataSetEntry();

    template <typename InIterT, typename OutIterT>
    DataSetEntry(InIterT inFirst, InIterT inLast,
                 OutIterT outFirst, OutIterT outLast);

    template <typename IterT>
    void setInputs(IterT first, IterT last);

    template <typename IterT>
    void setOutputs(IterT first, IterT last);

    std::size_t numOfInputs() const;

    std::size_t numOfOutputs() const;

    ValueT getField(std::size_t index) const;

    ValueT getInput(std::size_t index) const;

    ValueT getOutput(std::size_t index) const;

    //void resize(std::size_t ni, std::size_t no);

private:
    Container in_;
    Container out_;
}; // DataSetEntry


template <typename ValueT = fl::scalar>
class DataSet
{
private:
    typedef std::vector< DataSetEntry<ValueT> > EntryContainer;

    FL_MAKE_ITERATORS(public, EntryContainer, Entry, entry, entries_) 
    //FL_MAKE_STL_ITERATORS(public, EntryContainer, entries_) 


public:
    explicit DataSet(std::size_t ni = 0, std::size_t no = 0);

    void add(const DataSetEntry<ValueT>& entry);

    EntryIterator insert(EntryIterator pos, const DataSetEntry<ValueT>& entry);

    EntryIterator insert(ConstEntryIterator pos, const DataSetEntry<ValueT>& entry);

//  EntryIterator insert(ConstEntryIterator pos, const DataSetEntry<ValueT>& entry);

    EntryIterator erase(EntryIterator pos);

    EntryIterator erase(ConstEntryIterator pos);

    void set(const DataSetEntry<ValueT>& entry, std::size_t idx) const;

    const DataSetEntry<ValueT>& get(std::size_t idx) const;

	template <typename IterT>
	void setLabels(IterT first, IterT last);

	std::string getLabel(std::size_t idx) const;

	std::vector<std::string> labels() const;

    std::size_t size() const;

    bool empty() const;

    //void setNumOfInputs(std::size_t val);

    std::size_t numOfInputs() const;

    //void setNumOfOutputs(std::size_t val);

    std::size_t numOfOutputs() const;

    //void resize(std::size_t ni, std::size_t no);

    void clear();

	std::vector< std::vector<ValueT> > data() const;

private:
	void setDefaultLabels();


private:
    std::size_t ni_; ///< Number of inputs in each entry
    std::size_t no_; ///< Number of outputs in each entry
    EntryContainer entries_; ///< The collection of data entries
	std::vector<std::string> labels_; ///< The labels associated to data entry components
};


////////////////////////
// Template definitions
////////////////////////


////////////////
// DataSetEntry
////////////////


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

template <typename ValueT>
ValueT DataSetEntry<ValueT>::getInput(std::size_t index) const
{
    if (index >= in_.size())
    {
        FL_THROW2(std::invalid_argument, "Index is out-of-range");
    }

    return in_.at(index);
}

template <typename ValueT>
ValueT DataSetEntry<ValueT>::getOutput(std::size_t index) const
{
    if (index >= out_.size())
    {
        FL_THROW2(std::invalid_argument, "Index is out-of-range");
    }

    return out_.at(index);
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
//                                                                const DataSetEntry<ValueT>& entry)
//{
//  if (entry.numOfInputs() != ni_)
//  {
//      FL_THROW2(std::invalid_argument, "Unexpected number of inputs in the entry");
//  }
//  if (entry.numOfOutputs() != no_)
//  {
//      FL_THROW2(std::invalid_argument, "Unexpected number of outputs in the entry");
//  }
//
//  return entries_.insert(pos, entry);
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
	if (idx >= entries_.size())
	{
		FL_THROW2(std::invalid_argument, "Entry index is out-of-range");
	}

    return entries_[idx] = entry;
}

template <typename ValueT>
const DataSetEntry<ValueT>& DataSet<ValueT>::get(std::size_t idx) const
{
	if (idx >= entries_.size())
	{
		FL_THROW2(std::invalid_argument, "Entry index is out-of-range");
	}

    return entries_.at(idx);
}

template <typename ValueT>
template <typename IterT>
void DataSet<ValueT>::setLabels(IterT first, IterT last)
{
    labels_.clear();
    labels_.assign(first, last);
}

template <typename ValueT>
std::string DataSet<ValueT>::getLabel(std::size_t idx) const
{
	if (labels_.size() < (ni_+no_))
	{
		labels_ = this->setDefaultLabels();
	}

	if (idx >= labels_.size())
	{
		FL_THROW2(std::invalid_argument, "Label index is out-of-range");
	}

	return labels_.at(idx);
}

template <typename ValueT>
std::vector<std::string> DataSet<ValueT>::labels() const
{
	if (labels_.size() < (ni_+no_))
	{
		labels_ = this->setDefaultLabels();
	}

	return labels_;
}

template <typename ValueT>
std::size_t DataSet<ValueT>::size() const
{
    return entries_.size();
}

template <typename ValueT>
bool DataSet<ValueT>::empty() const
{
    return entries_.empty();
}

//template <typename ValueT>
//void DataSet<ValueT>::setNumOfInputs(std::size_t val)
//{
//  ni_ = val;
//}

template <typename ValueT>
std::size_t DataSet<ValueT>::numOfInputs() const
{
    return ni_;
}

//template <typename ValueT>
//void DataSet<ValueT>::setNumOfOutputs(std::size_t val)
//{
//  no_ = val;
//}

template <typename ValueT>
std::size_t DataSet<ValueT>::numOfOutputs() const
{
    return no_;
}

//template <typename ValueT>
//void DataSet<ValueT>::resize(std::size_t ni, std::size_t no)
//{
//  for (std::size_t i = 0,
//                   ni = entries_.size();
//       i < ni;
//       ++i)
//  {
//      entries_[i].resize(ni, no);
//  }
//}

template <typename ValueT>
void DataSet<ValueT>::clear()
{
    entries_.clear();
	labels_.clear();
}

template <typename ValueT>
void DataSet<ValueT>::setDefaultLabels()
{
	const std::size_t n = ni_+no_;
	const std::size_t width = std::floor(std::log10(n))+1;

	labels_.resize(n);
	for (std::size_t i = labels_.size(); i < n; ++i)
	{
		std::ostringstream oss;
		oss << "L" << std::setfill('0') << std::setw(width) << i;
		labels_[i] = oss.str();
	}
}

template <typename ValueT>
std::vector< std::vector<ValueT> > DataSet<ValueT>::data() const
{
	std::vector< std::vector<ValueT> > rawData;

	for (ConstEntryIterator it = entries_.begin(),
							endIt = entries_.end();
		 it != endIt;
		 ++it)
	{
		std::vector<ValueT> rawEntry;
//		for (typename DataSetEntry<ValueT>::ConstInputIterator entryIt = it->inputBegin(),
//															   entryEndIt = it->inputEnd();
//			 entryIt != entryEndIt;
//			 ++entryIt)
//		{
//			rawEntry.push_back(*entry
		rawEntry.assign(it->inputBegin(), it->inputEnd());
		rawEntry.insert(rawEntry.end(), it->outputBegin(), it->outputEnd());

		rawData.push_back(rawEntry);
	}

	return rawData;
}

} // Namespace fl

#endif // FL_DATASET_H
