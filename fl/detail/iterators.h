#ifndef FL_DETAIL_ITERATORS_H
#define FL_DETAIL_ITERATORS_H


#include <boost/type_erasure/any.hpp>
#include <boost/type_erasure/iterator.hpp>
#include <boost/type_erasure/operators.hpp>
#include <boost/type_erasure/same_type.hpp>
#include <boost/mpl/vector.hpp>
#include <fl/fuzzylite.h>


namespace fl { namespace detail {

struct iter_placeholder: boost::type_erasure::placeholder { };

#ifdef FL_CPP11

// We can use alias templates

//template <typename T>
//using ForwardIteratorImpl = boost::type_erasure::any<
//								boost::mpl::vector<
//								boost::type_erasure::copy_constructible<>,
//								boost::type_erasure::incrementable<>,
//								boost::type_erasure::dereferenceable<T>,
//								boost::type_erasure::equality_comparable<>
//							>>;
template <typename T>
using ForwardIteratorImpl = boost::type_erasure::any<
								boost::mpl::vector<
									boost::type_erasure::forward_iterator<iter_placeholder, const T&, std::ptrdiff_t, T>
								>,
								iter_placeholder
							>;

# define FL_ForwardIteratorType(t) fl::detail::ForwardIteratorImpl<t>

#else // FL_CPP1

// Use nested typedef to emulate alias templates...

template <typename T>
struct ForwardIteratorImpl
{
//	typedef boost::type_erasure::any<
//					boost::mpl::vector<
//						boost::type_erasure::copy_constructible<>,
//						boost::type_erasure::incrementable<>,
//						boost::type_erasure::dereferenceable<T>,
//						boost::type_erasure::equality_comparable<>
//					>
//				> type;
	typedef boost::type_erasure::any<
					boost::mpl::vector<
						boost::type_erasure::forward_iterator<iter_placeholder, const T&, std::ptrdiff_t, T>
					>,
					iter_placeholder
				> type;
};

# define FL_ForwardIteratorType(t) typename fl::detail::ForwardIteratorImpl<t>::type

#endif // FL_CPP11

}} // Namespace fl::detail

#endif // FL_DETAIL_ITERATORS_H
