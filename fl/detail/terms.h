/**
 * \file fl/detail/terms.h
 *
 * \brief Functionalities related to fuzzy terms
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

#ifndef FL_DETAIL_TERMS_H
#define FL_DETAIL_TERMS_H


#include <cstddef>
#include <fl/detail/math.h>
#include <fl/fuzzylite.h>
#include <fl/term/Term.h>
#include <fl/term/Bell.h>
#include <fl/term/Concave.h>
#include <fl/term/Constant.h>
#include <fl/term/Cosine.h>
#include <fl/term/Discrete.h>
#include <fl/term/Linear.h>
#include <fl/term/Ramp.h>
#include <fl/term/Sigmoid.h>
#include <fl/term/SShape.h>
#include <fl/term/Triangle.h>
#include <fl/term/ZShape.h>
#include <vector>


namespace fl { namespace detail {

std::vector<fl::scalar> GetTermParameters(const fl::Term* p_term);

template <typename IterT>
void SetTermParameters(fl::Term* p_term, IterT first, IterT last);

std::vector<fl::scalar> EvalBellTermDerivativeWrtParams(const fl::Bell& term, fl::scalar x);

std::vector<fl::scalar> EvalTermDerivativeWrtParams(const fl::Term* p_term, fl::scalar x);

template <typename IterT>
void SetTermParameters(fl::Term* p_term, IterT first, IterT last);

}} // Namespace fl::detail


#include "fl/detail/terms.tpp"


#endif // FL_DETAIL_TERMS_H
