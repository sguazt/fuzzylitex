#ifndef FL_DATASET_H
#define FL_DATASET_H


#include <cstddef>
#include <fl/commons.h>
#include <fl/fuzzylite.h>
#include <stdexcept>
#include <vector>


namespace fl {

template <typename ValueT = fl::scalar>
class DataSetEntry
{
	private: typedef std::vector<ValueT> Container;

	FL_MAKE_ITERATORS(public, Container, Input, input, in_) 
	FL_MAKE_ITERATORS(public, Container, Output, output, out_) 


	public: DataSetEntry()
	{
	}

	public: template <typename InIterT, typename OutIterT>
			DataSetEntry(InIterT inFirst, InIterT inLast,
						 OutIterT outFirst, OutIterT outLast)
	: in_(inFirst, inLast),
	  out_(outFirst, outLast)
	{
	}

	public: template <typename IterT>
			void setInputs(IterT first, IterT last)
	{
		in_.clear();
		in_.assign(first, last);
	}

	public: template <typename IterT>
			void setOutputs(IterT first, IterT last)
	{
		out_.clear();
		out_.assign(first, last);
	}

	public: std::size_t numOfInputs() const
	{
		return in_.size();
	}

	public: std::size_t numOfOutputs() const
	{
		return out_.size();
	}

	public: ValueT getField(std::size_t index) const
	{
		if (index >= (in_.size()+out_.size()))
		{
			FL_THROW2(std::invalid_argument, "Index is out-of-range");
		}

		return index < in_.size() ? in_.at(index)
								  : out_.at(index-in_.size());
	}


	private: Container in_;
	private: Container out_;
}; // DataSetEntry


template <typename ValueT = fl::scalar>
class DataSet
{
	private: typedef std::vector< DataSetEntry<ValueT> > EntryContainer;

	FL_MAKE_ITERATORS(public, EntryContainer, Entry, entry, entries_) 
	//FL_MAKE_STL_ITERATORS(public, EntryContainer, entries_) 


	public: explicit DataSet(std::size_t ni = 0, std::size_t no = 0)
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

	public: const DataSetEntry<ValueT>& get(std::size_t idx) const
	{
		return entries_.at(idx);
	}

	public: void get(const DataSetEntry<ValueT>& entry, std::size_t idx) const
	{
		return entries_[idx] = entry;
	}

	public: std::size_t size() const
	{
		return entries_.size();
	}

	public: void setNumOfInputs(std::size_t val)
	{
		ni_ = val;
	}

	public: std::size_t numOfInputs() const
	{
		return ni_;
	}

	public: void setNumOfOutputs(std::size_t val)
	{
		no_ = val;
	}

	public: std::size_t numOfOutputs() const
	{
		return no_;
	}


	private: std::size_t ni_; ///< Number of inputs in each entry
	private: std::size_t no_; ///< Number of outputs in each entry
	private: EntryContainer entries_;
};

} // Namespace fl

#endif // FL_DATASET_H
