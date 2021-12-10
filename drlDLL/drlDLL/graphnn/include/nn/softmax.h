#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

template<typename Dtype>
void SoftmaxDeriv(DTensor<CPU, Dtype>& dst, DTensor<CPU, Dtype>& cur_output, DTensor<CPU, Dtype>& cur_grad);

template<typename Dtype>
void SoftmaxDeriv(DTensor<GPU, Dtype>& dst, DTensor<GPU, Dtype>& cur_output, DTensor<GPU, Dtype>& cur_grad);


/**
 * @brief      the softmax activation operator
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class Softmax : public Factor
{
public:
	static std::string StrType()
	{
		return "Softmax";
	}

	using OutType = std::shared_ptr< DTensorVar<mode, Dtype> >;
	
	/**
	 * @brief      Creates an out variable.
	 *
	 * @return     return a tensor with same shape/type as input tensor
	 */
	OutType CreateOutVar()
	{
		auto out_name = this->name + ":out_0";
		return std::make_shared< DTensorVar<mode, Dtype> >(out_name);
	}

	/**
	 * @brief      constructor
	 *
	 * @param[in]  _name     The name
	 * @param[in]  _properr  whether propagate error
	 */
	Softmax(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						  std::vector< bool >& isConst, 
						  std::vector< std::shared_ptr<Variable> >& outputs) override;	
};

}
#endif