#ifndef FL_DETAIL_MATH_HPP
#define FL_DETAIL_MATH_HPP


#include <cstddef>
#include <vector>


namespace fl { namespace detail {

template <typename T>
T Sqr(T x)
{
	return x*x;
}

/// Returns evenly \a spaced values over the given interval [\a start, \a stop]. The end-point of the interval can be optionally excluded.
template <typename ValueT>
std::vector<ValueT> LinSpace(ValueT start, ValueT stop, std::size_t num, bool endPoint = true)
{
	std::vector<ValueT> res(num);

	if (endPoint)
	{
		if (num == 1)
		{
			res[0] = start;
		}
		else
		{
			const ValueT step = (stop-start)/(num-1);
			for (std::size_t i = 0; i < num; ++i)
			{
				res[i] = i*step+start;
			}
			res[num-1] = stop;
		}
	}
	else
	{
		const ValueT step = (stop-start)/num;
		for (std::size_t i = 0; i < num; ++i)
		{
			res[i] = i*step+start;
		}
	}

	return res;
}

}} // Namespace fl::detail


#endif // FL_DETAIL_MATH_HPP
