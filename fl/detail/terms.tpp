/**
 * \file fl/detail/terms.tpp
 *
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly.
 */

#ifndef FL_DETAIL_TERMS_TPP
#define FL_DETAIL_TERMS_TPP


namespace fl { namespace detail {

template <typename IterT>
void SetTermParameters(fl::Term* p_term, IterT first, IterT last)
{
	//FIXME: it would be a good idea to add a pure virtual method in fl::Term
	//       class that returns the vector of parameters, like:
	//         virtual std::vector<fl::scalar> getParameters() = 0;

	const std::vector<fl::scalar> params(first, last);

	if (dynamic_cast<fl::Bell*>(p_term))
	{
		fl::Bell* p_realTerm = dynamic_cast<fl::Bell*>(p_term);
		p_realTerm->setCenter(params[0]);
		p_realTerm->setWidth(params[1]);
		p_realTerm->setSlope(params[2]);
	}
	else if (dynamic_cast<fl::Concave*>(p_term))
	{
		fl::Concave* p_realTerm = dynamic_cast<fl::Concave*>(p_term);
		p_realTerm->setInflection(params[0]);
		p_realTerm->setEnd(params[1]);
	}
	else if (dynamic_cast<fl::Constant*>(p_term))
	{
		fl::Constant* p_realTerm = dynamic_cast<fl::Constant*>(p_term);
		p_realTerm->setValue(params[0]);
	}
	else if (dynamic_cast<fl::Cosine*>(p_term))
	{
		fl::Cosine* p_realTerm = dynamic_cast<fl::Cosine*>(p_term);
		p_realTerm->setCenter(params[0]);
		p_realTerm->setWidth(params[1]);
	}
	else if (dynamic_cast<fl::Discrete*>(p_term))
	{
		fl::Discrete* p_realTerm = dynamic_cast<fl::Discrete*>(p_term);
		const std::size_t np = params.size();
		std::vector<fl::Discrete::Pair> pairs;
		for (std::size_t p = 0; p < (np-1); p += 2)
		{
			pairs.push_back(fl::Discrete::Pair(params[p], params[p+1]));
		}
		p_realTerm->setXY(pairs);
	}
	else if (dynamic_cast<fl::Linear*>(p_term))
	{
		fl::Linear* p_realTerm = dynamic_cast<fl::Linear*>(p_term);
		p_realTerm->setCoefficients(params);
	}
	if (dynamic_cast<fl::Ramp*>(p_term))
	{
		fl::Ramp* p_realTerm = dynamic_cast<fl::Ramp*>(p_term);
		p_realTerm->setStart(params[0]);
		p_realTerm->setEnd(params[1]);
	}
	if (dynamic_cast<fl::Sigmoid*>(p_term))
	{
		fl::Sigmoid* p_realTerm = dynamic_cast<fl::Sigmoid*>(p_term);
		p_realTerm->setInflection(params[0]);
		p_realTerm->setSlope(params[1]);
	}
	else if (dynamic_cast<fl::SShape*>(p_term))
	{
		fl::SShape* p_realTerm = dynamic_cast<fl::SShape*>(p_term);
		p_realTerm->setStart(params[0]);
		p_realTerm->setEnd(params[1]);
	}
	else if (dynamic_cast<fl::Triangle*>(p_term))
	{
		fl::Triangle* p_realTerm = dynamic_cast<fl::Triangle*>(p_term);
		p_realTerm->setVertexA(params[0]);
		p_realTerm->setVertexB(params[1]);
		p_realTerm->setVertexC(params[2]);
	}
	else if (dynamic_cast<fl::ZShape*>(p_term))
	{
		fl::ZShape* p_realTerm = dynamic_cast<fl::ZShape*>(p_term);
		p_realTerm->setStart(params[0]);
		p_realTerm->setEnd(params[1]);
	}
}

}} // Namespace fl::detail

#endif // FL_DETAIL_TERMS_TPP
