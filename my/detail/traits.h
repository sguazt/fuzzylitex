#ifndef FL_DETAIL_TRAITS_H
#define FL_DETAIL_TRAITS_H


#include <algorithm>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/utility/enable_if.hpp>
//#include <cfloat>
#include <cmath>
#include <limits>


namespace fl { namespace detail {

/// See also:
/// - http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
/// - http://www.petebecker.com/js/js200012.html
/// - http://code.google.com/p/googletest/source/browse/trunk/include/gtest/internal/gtest-internal.h
/// - http://www.parashift.com/c++-faq-lite/newbie.html#faq-29.16
/// - http://adtmag.com/articles/2000/03/16/comparing-floats-how-to-determine-if-floating-quantities-are-close-enough-once-a-tolerance-has-been.aspx
/// - http://www.boost.org/doc/libs/1_47_0/libs/test/doc/html/utf/testing-tools/floating_point_comparison.html
/// - http://learningcppisfun.blogspot.com/2010/04/comparing-floating-point-numbers.html
/// .

template <typename T, typename Enable_ = void>
struct FloatTraits;

// Specialization for floating point types
template <typename T>
struct FloatTraits<T, typename ::boost::enable_if< ::boost::is_floating_point<T> >::type>
{
	/// Default tolerance for floating-point comparison.
	static const T tolerance;


	/**
	 * \brief x is approximately equal to y.
	 *
	 * Inspired by [1]:
	 * \f[
	 *  $x \approx y\,\text{ if and only if } |y-x|\le\epsilon\max(|x|,|y|)
	 * \f]
	 *
	 * References:
	 * -# Knuth, "The Art of Computer Programming: Vol.2" 3rd Ed, 1998, Sec. 4.2.2.
	 * .
	 */
	static bool ApproximatelyEqual(T x, T y, T tol = tolerance)
	{
		if (std::isnan(x) || std::isnan(y))
		{
			// According to IEEE, NaN are different event by itself
			return false;
		}
		return std::abs(x-y) <= (std::max(std::abs(x), std::abs(y))*tol);
	}

	/**
	 * \brief x is essentially equal to y.
	 *
	 * Inspired by [1]:
	 * \f[
	 *  $x \sim y\,\text{ if and only if } |y-x|\le\epsilon\min(|x|,|y|)
	 * \f]
	 *
	 * References:
	 * -# Knuth, "The Art of Computer Programming: Vol.2" 3rd Ed, 1998, Sec. 4.2.2.
	 * .
	 */
	static bool EssentiallyEqual(T x, T y, T tol = tolerance)
	{
		if (std::isnan(x) || std::isnan(y))
		{
			// According to IEEE, NaN are different event by itself
			return false;
		}
		return std::abs(x-y) <= (std::min(std::abs(x), std::abs(y))*tol);
	}

	/**
	 * \brief x is definitely less than y.
	 *
	 * Inspired by [1]:
	 * \f[
	 *  $x \prec y\,\text{ if and only if } y-x > \epsilon\max(|x|,|y|)
	 * \f]
	 *
	 * References:
	 * -# Knuth, "The Art of Computer Programming: Vol.2" 3rd Ed, 1998, Sec. 4.2.2.
	 * .
	 */
	static bool DefinitelyLess(T x, T y, T tol = tolerance)
	{
		return (y-x) > (std::max(std::abs(x), std::abs(y))*tol);
	}

	/**
	 * \brief x is definitely greater than y.
	 *
	 * Inspired by [1]:
	 * \f[
	 *  $x \succ y\,\text{ if and only if } x-y > \epsilon\max(|x|,|y|)
	 * \f]
	 *
	 * References:
	 * -# Knuth, "The Art of Computer Programming: Vol.2" 3rd Ed, 1998, Sec. 4.2.2.
	 * .
	 */
	static bool DefinitelyGreater(T x, T y, T tol = tolerance)
	{
		return (x-y) > (std::max(std::abs(x), std::abs(y))*tol);
	}

	static bool ApproximatelyLessEqual(T x, T y, T tol = tolerance)
	{
		return DefinitelyLess(x, y, tol) || ApproximatelyEqual(x, y, tol);
	}

	static bool EssentiallyLessEquall(T x, T y, T tol = tolerance)
	{
		return DefinitelyLess(x, y, tol) || EssentiallyEqual(x, y, tol);
	}

	static bool ApproximatelyGreaterEqual(T x, T y, T tol = tolerance)
	{
		return DefinitelyGreater(x, y, tol) || ApproximatelyEqual(x, y, tol);
	}

	static bool EssentiallyGreaterEquall(T x, T y, T tol = tolerance)
	{
		return DefinitelyGreater(x, y, tol) || EssentiallyEqual(x, y, tol);
	}

	static T DefinitelyMin(T x, T y, T tol = tolerance)
	{
		if (DefinitelyLess(x, y, tol))
		{
			return x;
		}
		return y;
	}

	static T Min(T x, T y)
	{
		return std::min(x, y);
	}

	static T DefinitelyMax(T x, T y, T tol = tolerance)
	{
		if (DefinitelyGreater(x, y, tol))
		{
			return x;
		}
		return y;
	}

	static T Max(T x, T y)
	{
		return std::max(x, y);
	}

	static T DefinitelyClamp(T x, T l, T h, T tol = tolerance)
	{
		return DefinitelyMin(h, DefinitelyMax(l, x));
	}

	static T Clamp(T x, T l, T h)
	{
		return Min(h, Max(l, x));
	}
}; // FloatTraits

template <typename T>
const T FloatTraits<T, typename boost::enable_if< boost::is_floating_point<T> >::type>::tolerance = static_cast<T>(100)*std::numeric_limits<T>::epsilon();

}} // Namespace fl::detail

#endif // FL_DETAIL_TRAITS_H
