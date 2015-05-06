/**
 * \file fl/macro.h
 *
 * \brief Header file for common macros
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

#ifndef FL_MACRO_H
#define FL_MACRO_H


#include <cassert>
#include <fl/fuzzylite.h>
#include <fl/Exception.h>
#include <sstream>


/// A macro just for expanding its argument (intended to use inside other macros)
#define FL_EXPAND__(x) x

//[FIXME]: the following part has been commented since strongly depends on the Boost library
/*
/// Pointer management macros
#ifdef FL_CPP11
# include <memory>
# define FL_shared_ptr std::shared_ptr
# define FL_make_shared std::make_shared
# define FL_function std::function
# ifndef FL_CPP14
#  if __cplusplus >= 201300L
#   define FL_CPP14 true
#  endif
# endif // CP_CPP14
# ifdef FL_CPP14
#  define FL_make_unique std::make_unique
# else
#  include <boost/make_unique.hpp>
#  define FL_make_unique boost::make_unique
# endif // FL_CPP14
#else // FL_CPP11
# include <boost/function.hpp>
# include <boost/shared_ptr.hpp>
# include <boost/make_shared.hpp>
# define FL_shared_ptr boost::shared_ptr
# define FL_make_shared boost::make_shared
# define FL_make_unique FL_unique_ptr
# define FL_function boost::function
#endif // FL_CPP11
*/
//[/FIXME]

#ifdef FL_CPP11
# define FL__FUNCTION__ __func__
#else // FL_CPP11
# include <boost/current_function.hpp>
# define FL__FUNCTION__ BOOST_CURRENT_FUNCTION
#endif // FL_CPP11

#ifndef NDEBUG
# define FL_DEBUG
# define FL_DEBUG_ASSERT(x) assert(x)
# define FL_DEBUG_TRACE(x) ::std::cerr << FL__FILE__ << "::" << FL__FUNCTION__ << "[" << __LINE__ << "]: " << FL_EXPAND__(x) << ::std::endl
#else // NDEBUG
# undef FL_DEBUG 
# define FL_DEBUG_ASSERT(x) FL_DEBUG_BEGIN \
                                assert(x); \
                            FL_DEBUG_END
# define FL_DEBUG_TRACE(x)  FL_DEBUG_BEGIN \
                                ::std::cerr << FL__FILE__ << "::" << FL__FUNCTION__ << "[" << __LINE__ << "]:" << FL_EXPAND__(x) << ::std::endl \
                            FL_DEBUG_END
#endif // NDEBUG

#define FL_THROW(m) throw fl::Exception(m, FL_AT)
#define FL_THROW2(e,m)  { \
                            std::ostringstream oss__; \
                            oss__ << (m) << std::endl << " {at " << FL__FILE__ << "::" << FL__FUNCTION__ << " [line: " << __LINE__ << "]}"; \
                            throw e(oss__.str()); \
                        }

#define FL_SUPPRESS_UNUSED_VARIABLE_WARNING(x) (void)(x);

#ifdef FL_CPP11
# define FL_IF_CPP11(t,e) t
#else
# define FL_IF_CPP11(t,e) e
#endif // FL_CPP11

/**
 * Macro to add iterator type definitions similarly to STL iterators.
 *
 * Macro parameters are the following:
 * - \c a: the access specifier,
 * - \c c: the container type,
 * - \c t: the type prefix
 * .
 * For instance:
 *  FL_MAKE_STL_ITERATOR_TYPES(public,std::vector<double>)
 * add the following type definitions:
 * <pre>
 * public: typedef typename std::vector<double>::iterator iterator;
 * public: typedef typename std::vector<double>::const_iterator const_iterator;
 * public: typedef typename std::vector<double>::reverse_iterator reverse_iterator;
 * public: typedef typename std::vector<double>::const_reverse_iterator const_reverse_iterator;
 * </pre>
 *
 * The intended use is inside a class or struct.
 */
#define FL_MAKE_STL_ITERATOR_TYPES(a,c) \
    a: typedef typename c::iterator iterator; \
    a: typedef typename c::const_iterator const_iterator; \
    a: typedef typename c::reverse_iterator reverse_iterator; \
    a: typedef typename c::const_reverse_iterator const_reverse_iterator;

/**
 * Macro to add iterator accessor definitions similarly to STL iterators.
 *
 * Macro parameters are the following:
 * - \c a: the access specifier,
 * - \c t: the type prefix,
 * - \c p: the prefix to add to each accessor method,
 * - \c v: the variable representing the real container
 * .
 * For instance:
 *  FL_MAKE_STL_ITERATOR_ACCESSORS(public,foos_)
 * add the following type definitions:
 * <pre>
 *  public: iterator begin() { return foos_.begin(); }
 *  public: iterator end() { return foos_.end(); }
 *  public: const_iterator begin() const { return foos_.begin(); }
 *  public: const_iterator end() const { return foos_.end(); }
 *  public: const_iterator cbegin() const { return foos_.cbegin(); }
 *  public: const_iterator cend() const { return foos_.cend(); }
 *  public: reverse_iterator rbegin() { return foos_.rbegin(); }
 *  public: reverse_iterator rend() { return foos_.rend(); }
 *  public: const_reverse_iterator rbegin() const { return foos_.rbegin(); }
 *  public: const_reverse_iterator rend() const { return foos_.rend(); }
 *  public: const_reverse_iterator crbegin() const { return foos_.crbegin(); }
 *  public: const_reverse_iterator crend() const { return foos_.crend(); }
 * </pre>
 *
 * The intended use is inside a class or struct.
 */
#define FL_MAKE_STL_ITERATOR_ACCESSORS(a,v) \
    a: iterator begin() { return (v).begin(); } \
    a: iterator end() { return (v).end(); } \
    a: const_iterator begin() const { return (v).begin(); } \
    a: const_iterator end() const { return (v).end(); } \
    a: const_iterator cbegin() const { return FL_IF_CPP11((v).cbegin(),(v).begin()); } \
    a: const_iterator cend() const { return FL_IF_CPP11((v).cend(),(v).end()); } \
    a: reverse_iterator rbegin() { return (v).rbegin(); } \
    a: reverse_iterator rend() { return (v).rend(); } \
    a: const_reverse_iterator rbegin() const { return (v).rbegin(); } \
    a: const_reverse_iterator rend() const { return (v).rend(); } \
    a: const_reverse_iterator crbegin() const { return FL_IF_CPP11((v).crbegin(),(v).rbegin()); } \
    a: const_reverse_iterator crend() const { return FL_IF_CPP11((v).crend(),(v).rend()); }

/// Macro that combines the above \c FL_MAKE_STL_ITERATOR_TYPES and \c FL_MAKE_STL_ITERATOR_ACCESSORS macros.
#define FL_STL_MAKE_ITERATORS(a,c,v) \
    FL_MAKE_STL_ITERATOR_TYPES(a,c) \
    FL_MAKE_STL_ITERATOR_ACCESSORS(a,v)

/**
 * Macro to add iterator type definitions similarly to STL iterators.
 *
 * Macro parameters are the following:
 * - \c a: the access specifier,
 * - \c c: the container type,
 * - \c t: the type prefix
 * .
 * For instance:
 *  FL_MAKE_ITERATOR_TYPES(public,std::vector<double>,Foobar)
 * add the following type definitions:
 * <pre>
 * public: typedef typename std::vector<double>::iterator FoobarIterator;
 * public: typedef typename std::vector<double>::const_iterator ConstFoobarIterator;
 * public: typedef typename std::vector<double>::reverse_iterator ReverseFoobarIterator;
 * public: typedef typename std::vector<double>::const_reverse_iterator ConstReverseFoobarIterator;
 * </pre>
 *
 * The intended use is inside a class or struct.
 */
#define FL_MAKE_ITERATOR_TYPES(a,c,t) \
    a: typedef typename c::iterator t##Iterator; \
    a: typedef typename c::const_iterator Const##t##Iterator; \
    a: typedef typename c::reverse_iterator Reverse##t##Iterator; \
    a: typedef typename c::const_reverse_iterator ConstReverse##t##Iterator;

/**
 * Macro to add iterator accessor definitions similarly to STL iterators.
 *
 * Macro parameters are the following:
 * - \c a: the access specifier,
 * - \c t: the type prefix,
 * - \c p: the prefix to add to each accessor method,
 * - \c v: the variable representing the real container
 * .
 * For instance:
 *  FL_MAKE_ITERATOR_ACCESSORS(public,Foobar,foobar,foos_)
 * add the following type definitions:
 * <pre>
 *  public: FoobarIterator foobarBegin() { return foos_.begin(); }
 *  public: FoobarIterator foobarEnd() { return foos_.end(); }
 *  public: ConstFoobarIterator foobarBegin() const { return foos_.begin(); }
 *  public: ConstFoobarIterator foobarEnd() const { return foos_.end(); }
 *  public: ConstFoobarIterator foobarCBegin() const { return foos_.cbegin(); }
 *  public: ConstFoobarIterator foobarCEnd() const { return foos_.cend(); }
 *  public: ReverseFoobarIterator foobarRBegin() { return foos_.rbegin(); }
 *  public: ReverseFoobarIterator foobarREnd() { return foos_.rend(); }
 *  public: ConstReverseFoobarIterator foobarRBegin() const { return foos_.rbegin(); }
 *  public: ConstReverseFoobarIterator foobarREnd() const { return foos_.rend(); }
 *  public: ConstReverseFoobarIterator foobarCRBegin() const { return foos_.crbegin(); }
 *  public: ConstReverseFoobarIterator foobarCREnd() const { return foos_.crend(); }
 * </pre>
 *
 * The intended use is inside a class or struct.
 */
#define FL_MAKE_ITERATOR_ACCESSORS(a,t,p,v) \
    a: t##Iterator p##Begin() { return (v).begin(); } \
    a: t##Iterator p##End() { return (v).end(); } \
    a: Const##t##Iterator p##Begin() const { return (v).begin(); } \
    a: Const##t##Iterator p##End() const { return (v).end(); } \
    a: Const##t##Iterator p##CBegin() const { return FL_IF_CPP11((v).cbegin(),(v).begin()); } \
    a: Const##t##Iterator p##CEnd() const { return FL_IF_CPP11((v).cend(),(v).end()); } \
    a: Reverse##t##Iterator p##RBegin() { return (v).rbegin(); } \
    a: Reverse##t##Iterator p##REnd() { return (v).rend(); } \
    a: ConstReverse##t##Iterator p##RBegin() const { return (v).rbegin(); } \
    a: ConstReverse##t##Iterator p##REnd() const { return (v).rend(); } \
    a: ConstReverse##t##Iterator p##CRBegin() const { return FL_IF_CPP11((v).crbegin(),(v).rbegin()); } \
    a: ConstReverse##t##Iterator p##CREnd() const { return FL_IF_CPP11((v).crend(),(v).rend()); }

/// Macro that combines the above \c FL_MAKE_ITERATOR_TYPES and \c FL_MAKE_ITERATOR_ACCESSORS macros.
#define FL_MAKE_ITERATORS(a,c,t,p,v) \
    FL_MAKE_ITERATOR_TYPES(a,c,t) \
    FL_MAKE_ITERATOR_ACCESSORS(a,t,p,v)

#endif // FL_MACRO_H
