#ifndef FL_DETAIL_DERIVATIVES_H
#define FL_DETAIL_DERIVATIVES_H


#include <cmath>
#include <my/commons.h>
#include <my/detail/math.h>
#include <fl/fuzzylite.h>
#include <fl/term/Bell.h>
#include <stdexcept>


namespace fl { namespace detail {

/**
 * Eval the derivative of the generalized bell function with respect to its parameters
 * \f{align}{
 *  \frac{\partial f(x,c,w,s)}{\partial x} &= -\frac{2s |\frac{x-c}{w}|^{2s-1}}{w (|\frac{x-c}{w}|^{2s}+1)^2},\\
 *  \frac{\partial f(x,c,w,s)}{\partial c} &=  \frac{2s |\frac{x-c}{w}|^{2s-1}}{w (|\frac{x-c}{w}|^{2s}+1)^2},\\
 *  \frac{\partial f(x,c,w,s)}{\partial w} &=  \frac{2s (x-c) |\frac{x-c}{w}|^{2s-1}}{w^2 (|\frac{x-c}{w}|^{2s}+1)^2},\\
 *  \frac{\partial f(x,c,w,s)}{\partial s} &= -\frac{2|\frac{x-c}{w}|^{2s} \log(|\frac{x-c}{w}|)}{(|\frac{x-c}{w}|^{2s}+1)^2}.
 * \f}
 *
 * Mathematica:
 *   f[x_, c_, w_, s_] := 1/(1 + Abs[(x - c)/w]^(2*s))
 *   D[f[x, c, w, s], {{x,c,w,s}}]
 */
std::vector<fl::scalar> EvalBellTermDerivativeWrtParams(const fl::Bell& term, fl::scalar x)
{
	const fl::scalar c = term.getCenter();
	const fl::scalar w = term.getWidth();
	const fl::scalar s = term.getSlope();

	const fl::scalar xn = (x-c)/w;
	const fl::scalar xnp = (xn != 0) ? std::pow(detail::Sqr(xn), s) : 0;
	const fl::scalar den = detail::Sqr(1+xnp);

	std::vector<fl::scalar> res(3);

	// Center parameter
	res[2] = (x != c)
			 ? 2.0*s*xnp/((x-c)*den)
			 : 0;
	// Width parameter
	res[0] = 2.0*s*xnp/(w*den);
	// Slope parameter
	res[1] = (x != c && x != (c+w))
			 ? -std::log(detail::Sqr(xn))*xnp/den
			 : 0;

	return res;
}

std::vector<fl::scalar> EvalTermDerivativeWrtParams(const fl::Term* p_term, fl::scalar x)
{
	if (dynamic_cast<const fl::Bell*>(p_term))
	{
		const fl::Bell* p_bell = dynamic_cast<const fl::Bell*>(p_term);
		return EvalBellTermDerivativeWrtParams(*p_bell, x);
	}

	FL_THROW2(std::runtime_error, "Derivative for term '" + p_term->className() + "' has not been implemented yet");
}

}} // Namespace fl::detail

#endif // FL_DETAIL_DERIVATIVES_H
