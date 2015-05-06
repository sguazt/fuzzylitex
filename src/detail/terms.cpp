/**
 * \file fl/detail/terms.cpp
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

#include <cmath>
#include <cstddef>
#include <fl/macro.h>
#include <fl/detail/math.h>
#include <fl/detail/terms.h>
#include <fl/fuzzylite.h>
#include <fl/term/Term.h>
#include <fl/term/Bell.h>
#include <fl/term/Concave.h>
#include <fl/term/Constant.h>
#include <fl/term/Cosine.h>
#include <fl/term/Discrete.h>
#include <fl/term/Gaussian.h>
#include <fl/term/GaussianProduct.h>
#include <fl/term/Linear.h>
#include <fl/term/Ramp.h>
#include <fl/term/Sigmoid.h>
#include <fl/term/SigmoidDifference.h>
#include <fl/term/SigmoidProduct.h>
#include <fl/term/SShape.h>
#include <fl/term/Triangle.h>
#include <fl/term/Trapezoid.h>
#include <fl/term/ZShape.h>
#include <stdexcept>
#include <vector>


namespace fl { namespace detail {

std::vector<fl::scalar> GetTermParameters(const fl::Term* p_term)
{
	//FIXME: it would be a good idea to add a pure virtual method in fl::Term
	//       class that returns the vector of parameters, like:
	//         virtual std::vector<fl::scalar> getParameters() = 0;

	std::vector<fl::scalar> params;

	if (dynamic_cast<const fl::Bell*>(p_term))
	{
		const fl::Bell* p_realTerm = dynamic_cast<const fl::Bell*>(p_term);
		params.push_back(p_realTerm->getCenter());
		params.push_back(p_realTerm->getWidth());
		params.push_back(p_realTerm->getSlope());
	}
	else if (dynamic_cast<const fl::Concave*>(p_term))
	{
		const fl::Concave* p_realTerm = dynamic_cast<const fl::Concave*>(p_term);
		params.push_back(p_realTerm->getInflection());
		params.push_back(p_realTerm->getEnd());
	}
	else if (dynamic_cast<const fl::Constant*>(p_term))
	{
		const fl::Constant* p_realTerm = dynamic_cast<const fl::Constant*>(p_term);
		params.push_back(p_realTerm->getValue());
	}
	else if (dynamic_cast<const fl::Cosine*>(p_term))
	{
		const fl::Cosine* p_realTerm = dynamic_cast<const fl::Cosine*>(p_term);
		params.push_back(p_realTerm->getCenter());
		params.push_back(p_realTerm->getWidth());
	}
	else if (dynamic_cast<const fl::Discrete*>(p_term))
	{
		const fl::Discrete* p_realTerm = dynamic_cast<const fl::Discrete*>(p_term);
		const std::vector<fl::Discrete::Pair> pairs = p_realTerm->xy();
		const std::size_t np = pairs.size();
		for (std::size_t p = 0; p < np; ++p)
		{
			params.push_back(pairs[p].first);
			params.push_back(pairs[p].second);
		}
	}
	else if (dynamic_cast<const fl::Gaussian*>(p_term))
	{
		const fl::Gaussian* p_realTerm = dynamic_cast<const fl::Gaussian*>(p_term);
		params.push_back(p_realTerm->getMean());
		params.push_back(p_realTerm->getStandardDeviation());
	}
	else if (dynamic_cast<const fl::GaussianProduct*>(p_term))
	{
		const fl::GaussianProduct* p_realTerm = dynamic_cast<const fl::GaussianProduct*>(p_term);
		params.push_back(p_realTerm->getMeanA());
		params.push_back(p_realTerm->getStandardDeviationA());
		params.push_back(p_realTerm->getMeanB());
		params.push_back(p_realTerm->getStandardDeviationB());
	}
	else if (dynamic_cast<const fl::Linear*>(p_term))
	{
		const fl::Linear* p_realTerm = dynamic_cast<const fl::Linear*>(p_term);
		params = p_realTerm->coefficients();
	}
	if (dynamic_cast<const fl::Ramp*>(p_term))
	{
		const fl::Ramp* p_realTerm = dynamic_cast<const fl::Ramp*>(p_term);
		params.push_back(p_realTerm->getStart());
		params.push_back(p_realTerm->getEnd());
	}
	if (dynamic_cast<const fl::Sigmoid*>(p_term))
	{
		const fl::Sigmoid* p_realTerm = dynamic_cast<const fl::Sigmoid*>(p_term);
		params.push_back(p_realTerm->getInflection());
		params.push_back(p_realTerm->getSlope());
	}
	if (dynamic_cast<const fl::SigmoidProduct*>(p_term))
	{
		const fl::SigmoidProduct* p_realTerm = dynamic_cast<const fl::SigmoidProduct*>(p_term);
		params.push_back(p_realTerm->getLeft());
		params.push_back(p_realTerm->getRising());
		params.push_back(p_realTerm->getFalling());
		params.push_back(p_realTerm->getRight());
	}
	else if (dynamic_cast<const fl::SShape*>(p_term))
	{
		const fl::SShape* p_realTerm = dynamic_cast<const fl::SShape*>(p_term);
		params.push_back(p_realTerm->getStart());
		params.push_back(p_realTerm->getEnd());
	}
	else if (dynamic_cast<const fl::Trapezoid*>(p_term))
	{
		const fl::Trapezoid* p_realTerm = dynamic_cast<const fl::Trapezoid*>(p_term);
		params.push_back(p_realTerm->getVertexA());
		params.push_back(p_realTerm->getVertexB());
		params.push_back(p_realTerm->getVertexC());
		params.push_back(p_realTerm->getVertexD());
	}
	else if (dynamic_cast<const fl::Triangle*>(p_term))
	{
		const fl::Triangle* p_realTerm = dynamic_cast<const fl::Triangle*>(p_term);
		params.push_back(p_realTerm->getVertexA());
		params.push_back(p_realTerm->getVertexB());
		params.push_back(p_realTerm->getVertexC());
	}
	else if (dynamic_cast<const fl::ZShape*>(p_term))
	{
		const fl::ZShape* p_realTerm = dynamic_cast<const fl::ZShape*>(p_term);
		params.push_back(p_realTerm->getStart());
		params.push_back(p_realTerm->getEnd());
	}

	return params;
}

std::vector<fl::scalar> EvalBellTermDerivativeWrtParams(const fl::Bell& term, fl::scalar x)
{
	/*
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

	const fl::scalar c = term.getCenter();
	const fl::scalar w = term.getWidth();
	const fl::scalar s = term.getSlope();

	const fl::scalar xn = (x-c)/w;
	const fl::scalar xnp = (xn != 0) ? std::pow(Sqr(xn), s) : 0;
	const fl::scalar den = Sqr(1+xnp);

	std::vector<fl::scalar> res(3);

	// Center parameter
	res[0] = (x != c)
			 ? 2.0*s*xnp/((x-c)*den)
			 : 0;
	// Width parameter
	res[1] = 2.0*s*xnp/(w*den);
	// Slope parameter
	res[2] = (x != c && x != (c+w))
			 ? -std::log(Sqr(xn))*xnp/den
			 : 0;

	return res;
}

std::vector<fl::scalar> EvalGaussianTermDerivativeWrtParams(const fl::Gaussian& term, fl::scalar x)
{
	/*
	 * Eval the derivative of the gaussian function with respect to its parameters
	 * \f{align}{
	 *  \frac{\partial f(x,c,s)}{\partial x} &= -\frac{(x-c) Exp[-\frac{(x-c)^2}{2 s^2}]}{s^2},\\
	 *  \frac{\partial f(x,c,s)}{\partial c} &=  \frac{(x-c) Exp[-\frac{(x-c)^2}{2 s^2}]}{s^2},\\
	 *  \frac{\partial f(x,c,s)}{\partial s} &=  \frac{(x-c)^2 Exp[-\frac{(x-c)^2}{2 s^2}]}{s^3}.
	 * \f}
	 *
	 * Mathematica:
	 *   f[x_, c_, s_] := Exp[-(x-c)^2/(2*s^2)]
	 *   D[f[x, c, s], {{x,c,s}}]
	 */

	const fl::scalar m = term.getMean();
	const fl::scalar sd = term.getStandardDeviation();

	const fl::scalar fx = term.membership(x);
	const fl::scalar sd2 = Sqr(sd);

	std::vector<fl::scalar> res(2);

	// Mean parameter
	res[0] = fx*(x-m)/sd2;
	// Standard deviation parameter
	res[1] = fx*Sqr(x-m)/(sd2*sd);

	return res;
}

std::vector<fl::scalar> EvalGaussianProductTermDerivativeWrtParams(const fl::GaussianProduct& term, fl::scalar x)
{
	/*
	 * Eval the derivative of the gaussian product function with respect to its parameters
	 * \f{align}{
	 *  \frac{\partial f(x,c1,s1,c2,s2)}{\partial x}  &= e^{-\frac{(x-c1)^2}{2 s1^2}-\frac{(x-c2)^2}{2 s2^2}} (-\frac{x-c1}{s1^2}-\frac{x-c2}{s2^2}),\\
	 *  \frac{\partial f(x,c1,s1,c2,s2)}{\partial c1} &= \frac{(x-c1) e^{-\frac{(x-c1)^2}{2 s1^2}-\frac{(x-c2)^2}{2 s2^2}}}{s1^2},\\
	 *  \frac{\partial f(x,c1,s1,c2,s2)}{\partial s1} &= \frac{(x-c1)^2 e^{-\frac{(x-c1)^2}{2 s1^2}-\frac{(x-c2)^2}{2 s2^2}}}{s1^3},\\
	 *  \frac{\partial f(x,c1,s1,c2,s2)}{\partial c2} &= \frac{(x-c2) e^{-\frac{(x-c1)^2}{2 s1^2}-\frac{(x-c2)^2}{2 s2^2}}}{s2^2},\\
	 *  \frac{\partial f(x,c1,s1,c2,s2)}{\partial s2} &= \frac{(x-c2)^2 e^{-\frac{(x-c1)^2}{2 s1^2}-\frac{(x-c2)^2}{2 s2^2}}}{s2^3}.
	 * \f}
	 *
	 * Mathematica:
	 *   f[x_, c1_, s1_, c2_, s2_] := Exp[-(x-c1)^2/(2*s1^2)]*Exp[-(x-c2)^2/(2*s2^2)]
	 *   D[f[x, c1, s1, c2, s2], {{x,c1,s1,c2,s2}}]
	 */

	const fl::scalar m1 = term.getMeanA();
	const fl::scalar sd1 = term.getStandardDeviationA();
	const fl::scalar m2 = term.getMeanB();
	const fl::scalar sd2 = term.getStandardDeviationB();

	const fl::scalar fx1 = (x >= m1) ? 1 : std::exp(-Sqr((x-m1)/sd1)/2.0);
	const fl::scalar fx2 = (x <= m2) ? 1 : std::exp(-Sqr((x-m2)/sd2)/2.0);
	const fl::scalar sd1sq = Sqr(sd1);
	const fl::scalar sd2sq = Sqr(sd2);
	const fl::scalar dfx1dm = (x >= m1) ? 0 : fx1*(x-m1)/sd1sq;
	const fl::scalar dfx2dm = (x <= m2) ? 0 : fx2*(x-m2)/sd2sq;
	const fl::scalar dfx1dsd = (x >= m1) ? 0 : fx1*Sqr(x-m1)/(sd1sq*sd1);
	const fl::scalar dfx2dsd = (x <= m2) ? 0 : fx2*Sqr(x-m2)/(sd2sq*sd2);

	std::vector<fl::scalar> res(4);

	// Mean parameter of the first gaussian
	res[0] = dfx1dm*fx2;
	// Standard deviation parameter of the first gaussian
	res[1] = dfx1dsd*fx2;
	// Mean parameter of the second gaussian
	res[2] = fx1*dfx2dm;
	// Standard deviation parameter of the second gaussian
	res[3] = fx1*dfx2dsd;

	return res;
}

std::vector<fl::scalar> EvalSigmoidTermDerivativeWrtParams(const fl::Sigmoid& term, fl::scalar x)
{
	/*
	 * Eval the derivative of the sigmoid function with respect to its parameters
	 * \f{align}{
	 *  \frac{\partial f(x,i,s)}{\partial x} &=  \frac{s e^{-s (x-i)}}{(e^{-s (x-i)}+1)^2},\\
	 *  \frac{\partial f(x,i,s)}{\partial i} &= -\frac{s e^{-s (x-i)}}{(e^{-s (x-i)}+1)^2},\\
	 *  \frac{\partial f(x,i,s)}{\partial s} &=  \frac{(x-i) e^{-s (x-i)}}{(e^{-s (x-i)}+1)^2}.
	 * \f}
	 *
	 * Mathematica:
	 *   f[x_, i_, s_] := 1/(1+Exp[-s*(x-i)])
	 *   D[f[x, i, s], {{x,i,s}}]
	 */

	const fl::scalar inflection = term.getInflection();
	const fl::scalar slope = term.getSlope();

	const fl::scalar fx = term.membership(x);

	std::vector<fl::scalar> res(2);

	// Inflection parameter
	res[0] = -fx*(1-fx)*slope;
	// Slope parameter
	res[1] = fx*(1-fx)*(x-inflection);

	return res;
}

std::vector<fl::scalar> EvalSigmoidDifferenceTermDerivativeWrtParams(const fl::SigmoidDifference& term, fl::scalar x)
{
	/*
	 * Eval the derivative of the sigmoid function with respect to its parameters
	 * \f{align}{
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial x}  &=  \frac{s1 e^{-s1 (x-i1)}}{(e^{-s1 (x-i1)}+1)^2}-\frac{s2 e^{-s2 (x-i2)}}{(e^{-s2 (x-i2)}+1)^2},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial i1} &= -\frac{s1 e^{-s1 (x-i1)}}{(e^{-s1 (x-i1)}+1)^2},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial s1} &=  \frac{(x-i1) e^{-s1 (x-i1)}}{(e^{-s1 (x-i1)}+1)^2},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial i2} &=  \frac{s2 e^{-s2 (x-i2)}}{(e^{-s2 (x-i2)}+1)^2},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial s2} &= -\frac{(x-i2) e^{-s2 (x-i2)}}{(e^{-s2 (x-i2)}+1)^2}.
	 * \f}
	 *
	 * Mathematica:
	 *   f[x_, i1_, s1_, i2_, s2_] := 1/(1+Exp[-s1*(x-i1)])-1/(1+Exp[-s2*(x-i2)])
	 *   D[f[x, i1, s1, i2, s2], {{x,i1,s1,i2,s2}}]
	 */

	const fl::scalar left = term.getLeft(); // inflection of the first sigmoid
	const fl::scalar rising = term.getRising(); // slope of the first sigmoid
	const fl::scalar falling = term.getFalling(); // slope of the second sigmoid
	const fl::scalar right = term.getRight(); // inflection of the second sigmoid

	const fl::scalar fx1 = 1.0/(1+std::exp(-rising*(x-left)));
	const fl::scalar fx2 = 1.0/(1+std::exp(-falling*(x-right)));
	const fl::scalar sgn = (fx1 >= fx2) ? 1 : -1;

	std::vector<fl::scalar> res(4);

	// Inflection parameter of the first sigmoid
	res[0] = -fx1*(1-fx1)*rising*sgn;
	// Slope parameter of the first sigmoid
	res[1] = fx1*(1-fx1)*(x-left)*sgn;
	// Slope parameter of the second sigmoid
	res[2] = -fx2*(1-fx2)*(x-right)*sgn;
	// Inflection parameter of the second sigmoid
	res[3] = fx2*(1-fx2)*falling*sgn;

	return res;
}

std::vector<fl::scalar> EvalSigmoidProductTermDerivativeWrtParams(const fl::SigmoidProduct& term, fl::scalar x)
{
	/*
	 * Eval the derivative of the sigmoid function with respect to its parameters
	 * \f{align}{
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial x}  &=  \frac{s1 e^{-s1 (x-i1)}}{(e^{-s1 (x-i1)}+1)^2 (e^{-s2 (x-i2)}+1)}+\frac{s2 e^{-s2 (x-i2)}}{(e^{-s1 (x-i1)}+1) (e^{-s2 (x-i2)}+1)^2},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial i1} &= -\frac{s1 e^{-s1 (x-i1)}}{(e^{-s1 (x-i1)}+1)^2 (e^{-s2 (x-i2)}+1)},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial s1} &=  \frac{(x-i1}) e^{-s1 (x-i1)}}{(e^{-s1 (x-i1)}+1)^2 (e^{-s2 (x-i2)}+1)},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial i2} &= -\frac{s2 e^{-s2 (x-i2)}}{(e^{-s1 (x-i1)}+1) (e^{-s2 (x-i2)}+1)^2},\\
	 *  \frac{\partial f(x,i1,s1,i2,s2)}{\partial s2} &=  \frac{(x-i2) e^{-s2 (x-i2)}}{(e^{-s1 (x-i1)}+1) (e^{-s2 (x-i2)}+1)^2}.
	 * \f}
	 *
	 *
	 * Mathematica:
	 *   f[x_, i1_, s1_, i2_, s2_] := 1/((1+Exp[-s1*(x-i1)])*(1+Exp[-s2*(x-i2)]))
	 *   D[f[x, i1, s1, i2, s2], {{x,i1,s1,i2,s2}}]
	 */

	const fl::scalar left = term.getLeft(); // inflection of the first sigmoid
	const fl::scalar rising = term.getRising(); // slope of the first sigmoid
	const fl::scalar falling = term.getFalling(); // slope of the second sigmoid
	const fl::scalar right = term.getRight(); // inflection of the second sigmoid

	const fl::scalar fx1 = 1.0/(1+std::exp(-rising*(x-left)));
	const fl::scalar fx2 = 1.0/(1+std::exp(-falling*(x-right)));

	std::vector<fl::scalar> res(4);

	// Inflection parameter of the first sigmoid
	res[0] = -fx1*(1-fx1)*rising*fx2;
	// Slope parameter of the first sigmoid
	res[1] = fx1*(1-fx1)*(x-left)*fx2;
	// Slope parameter of the second sigmoid
	res[2] = fx1*fx2*(1-fx2)*(x-right);
	// Inflection parameter of the second sigmoid
	res[3] = -fx1*fx2*(1-fx2)*falling;

	return res;
}

std::vector<fl::scalar> EvalTrapezoidTermDerivativeWrtParams(const fl::Trapezoid& term, fl::scalar x)
{
	const fl::scalar a = term.getVertexA(); // left feet of the trapezoid
	const fl::scalar b = term.getVertexB(); // left shoulder of the trapezoid
	const fl::scalar c = term.getVertexC(); // right shoulder of the trapezoid
	const fl::scalar d = term.getVertexD(); // right feet of the trapezoid

	const fl::scalar fx1 = (b <= x) ? 1 : ((x < a) ? 0 : ((a != b) ? (x-a)/(b-a) : 0));
	const fl::scalar fx2 = (x <= c) ? 1 : ((d < x) ? 0 : ((c != d) ? (d-x)/(d-c) : 0));

	std::vector<fl::scalar> res(4);

	if (fx1 < fx2)
	{
		// Vertex A
		res[0] = (a <= x && x <= b)
				 ? -(b-x)/Sqr(b-a)
				 : 0;
		// Vertex B
		res[1] = (a <= x && x <= b)
				 ? -(x-a)/Sqr(b-a)
				 : 0;
		// Vertex C
		res[2] = 0;
		// Vertex D
		res[3] = 0;
	}
	else
	{
		// Vertex A
		res[0] = 0;
		// Vertex B
		res[1] = 0;
		// Vertex C
		res[2] = (c <= x && x <= d)
				 ? (d-x)/Sqr(d-c)
				 : 0;
		// Vertex D
		res[3] = (c <= x && x <= d)
				 ? (x-c)/Sqr(d-c)
				 : 0;
	}

	return res;
}

std::vector<fl::scalar> EvalTriangleTermDerivativeWrtParams(const fl::Triangle& term, fl::scalar x)
{
	const fl::scalar a = term.getVertexA(); // left feet of the triangle
	const fl::scalar b = term.getVertexB(); // peek of the triangle
	const fl::scalar c = term.getVertexC(); // right feet of the triangle

	std::vector<fl::scalar> res(3);

	// Vertex A
	res[0] = (a <= x && x <= b)
			 ? -(b-x)/Sqr(b-a)
			 : 0;
	// Vertex B
	res[1] = (a <= x && x <= b)
			 ? -(x-a)/Sqr(b-a)
			 : ((b <= x && x <= c)
			 	? -(c-x)/Sqr(c-b)
			 	: 0);
	// Vertex C
	res[2] = (b <= x && x <= c)
			 ? -(x-b)/Sqr(c-b)
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
	else if (dynamic_cast<const fl::Gaussian*>(p_term))
	{
		const fl::Gaussian* p_gauss = dynamic_cast<const fl::Gaussian*>(p_term);
		return EvalGaussianTermDerivativeWrtParams(*p_gauss, x);
	}
	else if (dynamic_cast<const fl::GaussianProduct*>(p_term))
	{
		const fl::GaussianProduct* p_gaussProd = dynamic_cast<const fl::GaussianProduct*>(p_term);
		return EvalGaussianProductTermDerivativeWrtParams(*p_gaussProd, x);
	}
	else if (dynamic_cast<const fl::Sigmoid*>(p_term))
	{
		const fl::Sigmoid* p_sig = dynamic_cast<const fl::Sigmoid*>(p_term);
		return EvalSigmoidTermDerivativeWrtParams(*p_sig, x);
	}
	else if (dynamic_cast<const fl::SigmoidDifference*>(p_term))
	{
		const fl::SigmoidDifference* p_sigDiff = dynamic_cast<const fl::SigmoidDifference*>(p_term);
		return EvalSigmoidDifferenceTermDerivativeWrtParams(*p_sigDiff, x);
	}
	else if (dynamic_cast<const fl::SigmoidProduct*>(p_term))
	{
		const fl::SigmoidProduct* p_sigProd = dynamic_cast<const fl::SigmoidProduct*>(p_term);
		return EvalSigmoidProductTermDerivativeWrtParams(*p_sigProd, x);
	}
	else if (dynamic_cast<const fl::Trapezoid*>(p_term))
	{
		const fl::Trapezoid* p_trap = dynamic_cast<const fl::Trapezoid*>(p_term);
		return EvalTrapezoidTermDerivativeWrtParams(*p_trap, x);
	}
	else if (dynamic_cast<const fl::Triangle*>(p_term))
	{
		const fl::Triangle* p_tri= dynamic_cast<const fl::Triangle*>(p_term);
		return EvalTriangleTermDerivativeWrtParams(*p_tri, x);
	}

	FL_THROW2(std::runtime_error, "Derivative for term '" + p_term->className() + "' has not been implemented yet");
}

}} // Namespace fl::detail
