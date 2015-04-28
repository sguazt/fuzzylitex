/**
 * \file fl/detail/random.h
 *
 * \brief Utilities for pseudo random number generation
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

#ifndef FL_DETAIL_RANDOM_H
#define FL_DETAIL_RANDOM_H


//#include <cstdlib>
#include <fl/fuzzylite.h>
#ifdef FL_CPP11
# include <random>
#else
# include <boost/random.hpp>
#endif // FL_CPP11


namespace fl { namespace detail {

// Poor man random number generation
//double RandUnif01(unsigned int& state)
//{
//	return std::rand_r(&state)/(static_cast<double>(RAND_MAX)+1.0);
//}
//
//int RandUnif(int from, int thru)
//{
//	static unsigned int state = 1;
//	return RandUnif01(&state)*(thru-from+1) + min;
//}
//
//double RandUnif(double from, double upto)
//{
//	static unsigned int state = 1;
//	return RandUnif01(&state)*(upto-from) + min;
//}

#ifdef FL_CPP11

inline
std::default_random_engine& GlobalUrng()
{
    static std::default_random_engine u{};
    return u;
}

inline
int RandUnif(int from, int thru)
{
    static std::uniform_int_distribution<> d {};
    using parm_t = decltype(d)::param_type;
    return d(GlobalUrng(), parm_t{from, thru});
}

template <typename EngineT>
int RandUnif(int from, int thru, EngineT& eng)
{
    static std::uniform_int_distribution<> d {};
    using parm_t = decltype(d)::param_type;
    return d(eng, parm_t{from, thru});
}

template <typename RealT>
RealT RandUnif(RealT from, RealT upto)
{
    static std::uniform_real_distribution<RealT> d {};
    using parm_t = decltype(d)::param_type;
    return d(GlobalUrng(), parm_t{from, upto});
}

template <typename RealT, typename EngineT>
RealT RandUnif(RealT from, RealT upto, EngineT& eng)
{
    static std::uniform_real_distribution<RealT> d {};
    using parm_t = decltype(d)::param_type;
    return d(eng, parm_t{from, upto});
}

template <typename RealT>
RealT RandNormal(RealT mean, RealT sd)
{
    static std::normal_distribution<RealT> d {};
    using parm_t = decltype(d)::param_type;
    return d(GlobalUrng(), parm_t{mean, sd});
}

template <typename RealT, typename EngineT>
RealT RandNormal(RealT mean, RealT sd, EngineT& eng)
{
    static std::normal_distribution<RealT> d {};
    using parm_t = decltype(d)::param_type;
    return d(eng, parm_t{mean, sd});
}

#else // FL_CPP11

inline
boost::random::mt19937& GlobalUrng()
{
    static boost::random::mt19937 u;
    return u;
}

inline
int RandUnif(int from, int thru)
{
    static boost::random::uniform_int_distribution<> d;
    return d(GlobalUrng(), typename boost::random::uniform_int_distribution<>::param_type(from, thru));
}

template <typename EngineT>
int RandUnif(int from, int thru, EngineT& eng)
{
    static boost::random::uniform_int_distribution<> d;
    return d(eng, typename boost::random::uniform_int_distribution<>::param_type(from, thru));
}

template <typename RealT>
RealT RandUnif(RealT from, RealT upto)
{
    static boost::random::uniform_real_distribution<RealT> d;
    return d(GlobalUrng(), typename boost::random::uniform_real_distribution<RealT>::param_type(from, upto));
}

template <typename RealT, typename EngineT>
RealT RandUnif(RealT from, RealT upto, EngineT& eng)
{
    static boost::random::uniform_real_distribution<RealT> d;
    return d(eng, typename boost::random::uniform_real_distribution<RealT>::param_type(from, upto));
}

template <typename RealT>
RealT RandNormal(RealT mean, RealT sd)
{
    static boost::random::normal_distribution<RealT> d;
    return d(GlobalUrng(), typename boost::random::normal_distribution<RealT>::param_type(mean, sd));
}

template <typename RealT, typename EngineT>
RealT RandNormal(RealT mean, RealT sd, EngineT& eng)
{
    static boost::random::normal_distribution<RealT> d;
    return d(eng, typename boost::random::normal_distribution<RealT>::param_type(mean, sd));
}

#endif // FL_CPP11

}} // Namespace fl::detail

#endif // FL_DETAIL_RANDOM_H
