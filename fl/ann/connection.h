#ifndef FL_ANN_CONNECTION_H
#define FL_ANN_CONNECTION_H


#include <fl/fuzzylite.h>
//#include <limits>


namespace fl { namespace ann {

// Forward declarations
template <typename T> class Neuron;


template <typename ValueT>
class Connection
{
	public: Connection()
	: p_from_(fl::null),
	  p_to_(fl::null),
	  w_(0)
	{
	}

	public: Connection(Neuron<ValueT>* p_from, Neuron<ValueT>* p_to, ValueT weight = 0)
	: p_from_(p_from),
	  p_to_(p_to),
	  w_(weight)
	{
	}

	public: virtual ~Connection() { }

	/// Resets the weight of this connection
	public: void reset()
	{
		w_ = 0;
	}

	/// Sets the neuron at the source
	public: void setFromNeuron(Neuron<ValueT>* p_neuron)
	{
		p_from_ = p_neuron;
	}

	/// Returns the neuron at the source
	public: Neuron<ValueT>* getFromNeuron() const
	{
		return p_from_;
	}

	/// Sets the neuron at the destination
	public: void setToNeuron(Neuron<ValueT>* p_neuron)
	{
		p_to_ = p_neuron;
	}

	/// Returns the neuron at the destination
	public: Neuron<ValueT>* getToNeuron() const
	{
		return p_to_;
	}

	/// Sets the weight
	public: void setWeight(ValueT v)
	{
		w_ = v;
	}

	/// Returns the weight
	public: ValueT getWeight() const
	{
		return w_;
	}


	private: Neuron<ValueT>* p_from_; ///< A pointer to the neuron at the source
	private: Neuron<ValueT>* p_to_; ///< A pointer to the neuron at the destination
	private: ValueT w_; ///< The weight
}; // Connection

}} // Namespace fl::ann

#endif // FL_ANN_CONNECTION_H
