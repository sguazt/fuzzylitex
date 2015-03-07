#ifndef FL_DATASET_H
#define FL_DATASET_H


#include <cstddef>
#include <my/commons.h>
#include <stdexcept>
#include <vector>


namespace fl {

template <typename ValueT>
class DataSetEntry
{
	private: typedef std::vector<ValueT> Container;

	FL_MAKE_ITERATORS(public, Container, Input, input, in_) 
	FL_MAKE_ITERATORS(public, Container, Output, output, out_) 


	public: template <typename InIterT, typename OutIterT>
			DataSetEntry(InIterT inFirst, InIterT inLast,
						 OutIterT outFirst, OutIterT outLast)
	: in_(inFirst, inLast),
	  out_(outFirst, outLast)
	{
	}

	public: std::size_t numOfInputs() const
	{
		return in_.size();
	}

	public: std::size_t numOfOutputs() const
	{
		return out_.size();
	}


	private: Container in_;
	private: Container out_;
}; // DataSetEntry


template <typename ValueT>
class DataSet
{
	private: typedef std::vector< DataSetEntry<ValueT> > EntryContainer;

	FL_MAKE_ITERATORS(public, EntryContainer, Entry, entry, entries_) 
	//FL_MAKE_STL_ITERATORS(public, EntryContainer, entries_) 


	public: DataSet(std::size_t ni, std::size_t no)
	: ni_(ni),
	  no_(no)
	{
	}

	public: void add(const DataSetEntry<ValueT>& entry)
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

	public: EntryIterator insert(EntryIterator pos, const DataSetEntry<ValueT>& entry)
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

	public: EntryIterator insert(ConstEntryIterator pos, const DataSetEntry<ValueT>& entry)
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

//	public: EntryIterator insert(ConstEntryIterator pos, const DataSetEntry<ValueT>& entry)
//	{
//		if (entry.numOfInputs() != ni_)
//		{
//			FL_THROW2(std::invalid_argument, "Unexpected number of inputs in the entry");
//		}
//		if (entry.numOfOutputs() != no_)
//		{
//			FL_THROW2(std::invalid_argument, "Unexpected number of outputs in the entry");
//		}
//
//		return entries_.insert(pos, entry);
//	}

	public: EntryIterator erase(EntryIterator pos)
	{
		return entries_.erase(pos);
	}

	public: EntryIterator erase(ConstEntryIterator pos)
	{
		return entries_.erase(pos);
	}

	public: std::size_t size() const
	{
		return entries_.size();
	}


	private: std::size_t ni_; ///< Number of inputs in each entry
	private: std::size_t no_; ///< Number of outputs in each entry
	private: EntryContainer entries_;
};

} // Namespace fl

#endif // FL_DATASET_H
