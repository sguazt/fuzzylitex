#ifndef FL_DETAIL_KALMAN_H
#define FL_DETAIL_KALMAN_H


#include <algorithm>
#include <cstddef>
#include <cmath>
#include <fl/detail/math.h>
#include <fl/fuzzylite.h>
#include <vector>


namespace fl { namespace detail {

template <typename ValueT>
class KalmanFilter
{
public:
	KalmanFilter(std::size_t p, std::size_t nin, std::size_t nout, ValueT lambda = 0.98)
	: p_(p),
	  in_n_(nin),
	  out_n_(nout),
	  lambda_(lambda),
	  S_(0),
	  P_(0),
	  a_(0),
	  b_(0),
	  a_t_(0),
	  b_t_(0),
	  tmp1_(0),
	  tmp2_(0),
	  tmp3_(0),
	  tmp4_(0),
	  tmp5_(0),
	  tmp6_(0),
	  tmp7_(0)
	{
		allocateMemory();
	}

	~KalmanFilter()
	{
		deallocateMemory();
	}

	void reset(ValueT alpha = 1e+6)
	{
		reallocateMemory();

		// reset S and P
		for (std::size_t i = 0; i < in_n_; ++i)
		{
			for (std::size_t j = 0; j < out_n_; ++j)
			{
				P_[i][j] = 0;
			}
		}
		for (std::size_t i = 0; i < in_n_; ++i)
		{
			for (std::size_t j = 0; j < in_n_; ++j)
			{
				if (i == j)
				{
					S_[i][j] = alpha; 
				}
				else
				{
					S_[i][j] = 0;
				}
			}
		}
	}

	template <typename InIterT, typename OutIterT>
	std::vector<ValueT> estimate(InIterT inFirst, InIterT inLast, OutIterT outFirst, OutIterT outLast)
	{
		ValueT* in = new ValueT[in_n_];
		std::copy(inFirst, inLast, in);

		ValueT* out = new ValueT[out_n_];
		std::copy(outFirst, outLast, out);

		std::vector<ValueT> ret;

		ret = this->estimate(in, out);

		delete[] out;
		delete[] in;

		return ret;
	}

	std::vector<ValueT> estimate(ValueT* in, ValueT* out)
	{
		std::vector<ValueT> ret(out_n_);

		for (std::size_t i = 0; i < in_n_; ++i)
		{
			a_[i][0] = in[i];
		}

		for (std::size_t i = 0; i < out_n_; ++i)
		{
			b_[i][0] = out[i];
		}

		transposeM(a_, in_n_, 1, a_t_);
		transposeM(b_, out_n_, 1, b_t_);

		/* recursive formulas for S, covariance matrix */
		anfisMtimesM(S_, a_, in_n_, in_n_, 1, tmp1_);
		anfisMtimesM(a_t_, tmp1_, 1, in_n_, 1, tmp2_);
		ValueT denom = lambda_ + tmp2_[0][0];
		anfisMtimesM(a_t_, S_, 1, in_n_, in_n_, tmp3_);
		anfisMtimesM(tmp1_, tmp3_, in_n_, 1, in_n_, tmp4_);
		anfisStimesM(1/denom, tmp4_, in_n_, in_n_, tmp4_);
		anfisMminusM(S_, tmp4_, in_n_, in_n_, S_);
		anfisStimesM(1/lambda_, S_, in_n_, in_n_, S_);
//std::cerr << "KAL - Covariance Matrix ="; fl::detail::MatrixOutput(std::cerr, S_, in_n_, in_n_); std::cerr << std::endl;//XXX

		// Compute the output estimate
		anfisMtimesM(a_t_, P_, 1, in_n_, out_n_, tmp5_);
		std::copy(tmp5_[0], tmp5_[0]+out_n_, ret.begin());

		/* recursive formulas for P, the estimated parameter matrix */
		anfisMtimesM(a_t_, P_, 1, in_n_, out_n_, tmp5_);
		anfisMminusM(b_t_, tmp5_, 1, out_n_, tmp5_);
		anfisMtimesM(a_, tmp5_, in_n_, 1, out_n_, tmp6_);
		anfisMtimesM(S_, tmp6_, in_n_, in_n_, out_n_, tmp7_);
		anfisMplusM(P_, tmp7_, in_n_, out_n_, P_);

		return ret;
	}

	std::vector< std::vector<ValueT> > getEstimatedParameters() const
	{
		std::vector< std::vector<ValueT> > ret(in_n_);
		for (std::size_t i = 0; i < in_n_; ++i)
		{
			ret[i].resize(out_n_);
			std::copy(P_[i], P_[i]+out_n_, ret[i].begin());
		}
		return ret;
	}

	std::vector< std::vector<ValueT> > getCovarianceInverse() const
	{
		std::vector< std::vector<ValueT> > ret(in_n_);
		for (std::size_t i = 0; i < in_n_; ++i)
		{
			ret[i].resize(in_n_);
			std::copy(S_[i], S_[i]+in_n_, ret[i].begin());
		}
		return ret;
	}

	std::vector<ValueT> getRegressor() const
	{
		std::vector<ValueT> ret(in_n_);
		for (std::size_t i = 0; i < in_n_; ++i)
		{
			ret[i] = a_[i][0];
		}
		return ret;
	}

	void setModelOrder(std::size_t p)
	{
		p_ = p+1;
	}

	std::size_t getModelOrder() const
	{
		if (p_ == 0)
		{
			return 0;
		}
		return p_-1;
	}

	void setInputDimension(std::size_t n)
	{
		in_n_ = n;
	}

	std::size_t getInputDimension() const
	{
		return in_n_;
	}

	void setOutputDimension(std::size_t n)
	{
		out_n_ = n;
	}

	std::size_t getOutputDimension() const
	{
		return out_n_;
	}

	void setForgettingFactor(ValueT value)
	{
		lambda_ = value;
	}

	ValueT getForgettingFactor() const
	{
		return lambda_;
	}


private:
	void allocateMemory()
	{
		if (in_n_ > 0 && out_n_ > 0)
		{
			S_ = createMatrix(in_n_, in_n_);
			P_ = createMatrix(in_n_, out_n_);
			a_ = createMatrix(in_n_, 1);
			b_ = createMatrix(out_n_, 1);
			a_t_ = createMatrix(1, in_n_);
			b_t_ = createMatrix(1, out_n_);
			tmp1_ = createMatrix(in_n_, 1);
			tmp2_ = createMatrix(1, 1);
			tmp3_ = createMatrix(1, in_n_);
			tmp4_ = createMatrix(in_n_, in_n_);
			tmp5_ = createMatrix(1, out_n_);
			tmp6_ = createMatrix(in_n_, out_n_);
			tmp7_ = createMatrix(in_n_, out_n_);
		}
	}

	void deallocateMemory()
	{
		destroyMatrix(S_, in_n_, in_n_);
		S_ = 0;
		destroyMatrix(P_, in_n_, out_n_);
		P_ = 0;
		destroyMatrix(a_, in_n_, 1);
		a_ = 0;
		destroyMatrix(b_, out_n_, 1);
		b_ = 0;
		destroyMatrix(a_t_, 1, in_n_);
		a_t_ = 0;
		destroyMatrix(b_t_, 1, out_n_);
		b_t_ = 0;
		destroyMatrix(tmp1_, in_n_, 1);
		tmp1_ = 0;
		destroyMatrix(tmp2_, 1, 1);
		tmp2_ = 0;
		destroyMatrix(tmp3_, 1, in_n_);
		tmp3_ = 0;
		destroyMatrix(tmp4_, in_n_, in_n_);
		tmp4_ = 0;
		destroyMatrix(tmp5_, 1, out_n_);
		tmp5_ = 0;
		destroyMatrix(tmp6_, in_n_, out_n_);
		tmp6_ = 0;
		destroyMatrix(tmp7_, in_n_, out_n_);
		tmp7_ = 0;
	}

	void reallocateMemory()
	{
		deallocateMemory();
		allocateMemory();
	}

	static ValueT** createMatrix(std::size_t nrows, std::size_t ncols)
	{
		ValueT** tmp = new ValueT*[nrows];
		for (std::size_t i = 0; i < nrows; ++i)
		{
			tmp[i] = new ValueT[ncols];
		}
		return tmp;
	}

	static void destroyMatrix(ValueT** m, std::size_t nrows, std::size_t ncols)
	{
		(void) ncols;
		if (m)
		{
			for (std::size_t i = 0; i < nrows; ++i)
			{
				if (m[i])
				{
					delete[] m[i];
				}
			}
			delete[] m;
		}
	}

	/* matrix plus matrix */
	static void anfisMplusM(ValueT **m1, ValueT **m2, std::size_t row, std::size_t col, ValueT **out)
	{
		for (std::size_t i = 0; i < row; ++i)
			for (std::size_t j = 0; j < col; ++j)
				out[i][j] = m1[i][j] + m2[i][j];
	}

	/* matrix minus matrix */
	static void anfisMminusM(ValueT **m1, ValueT **m2, std::size_t row, std::size_t col, ValueT **out)
	{
		for (std::size_t i = 0; i < row; ++i)
			for (std::size_t j = 0; j < col; ++j)
				out[i][j] = m1[i][j] - m2[i][j];
	}

	/* matrix times matrix */
	static void anfisMtimesM(ValueT **m1, ValueT **m2, std::size_t row1, std::size_t col1, std::size_t col2, ValueT **out)
	{
		for (std::size_t i = 0; i < row1; ++i)
			for (std::size_t j = 0; j < col2; ++j) {
				out[i][j] = 0;
				for (std::size_t k = 0; k < col1; ++k)
					out[i][j] += m1[i][k]* m2[k][j];
			}
	}

	/* scalar times matrix */
	static void anfisStimesM(ValueT c, ValueT **m, std::size_t row, std::size_t col, ValueT **out)
	{
		for (std::size_t i = 0; i < row; ++i)
			for (std::size_t j = 0; j < col; ++j)
				out[i][j] = c*m[i][j];
	}

	/* matrix transpose */
	static void transposeM(ValueT **m, std::size_t row, std::size_t col, ValueT **m_t)
	{
		for (std::size_t i = 0; i < row; ++i)
			for (std::size_t j = 0; j < col; ++j)
				m_t[j][i] = m[i][j];
	}

	/* matrix L-2 norm */
	static ValueT matrixNorm(ValueT **m, std::size_t row, std::size_t col)
	{
		ValueT total = 0;

		for (std::size_t i = 0; i < col; ++i)
			for (std::size_t j = 0; j < row; ++j)
				total += m[i][j]*m[i][j];
		return(std::sqrt(total));
	}


private:
	std::size_t p_; ///< Model order
	std::size_t in_n_; ///< Number of inputs
	std::size_t out_n_; ///< Number of outputs
	ValueT lambda_; ///< Forgetting factor
	ValueT** S_; ///< Inverse covariance matrix
	ValueT** P_; ///< Parameters matrix
	ValueT** a_; ///< Regressor vector
	ValueT** b_; ///< Output vector
	ValueT** a_t_; ///< Transpose of regressor vector
	ValueT** b_t_; ///< Transpose of output vector
	ValueT** tmp1_;
	ValueT** tmp2_;
	ValueT** tmp3_;
	ValueT** tmp4_;
	ValueT** tmp5_;
	ValueT** tmp6_;
	ValueT** tmp7_;
}; // KalmanFilter

}} // Namespace fl::detail

#endif // FL_DETAIL_KALMAN_H
