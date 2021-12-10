#ifndef EXP_H
#define EXP_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

#include <memory>

namespace gnn
{

/**
 * @brief      the exp operator
 *
 * @tparam     mode   { CPU/GPU }
 * @tparam     Dtype  { float/double }
 */
template<typename mode, typename Dtype>
class Exp : public Factor
{
public:
	static std::string StrType()
	{
		return "Exp";
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
	Exp(std::string _name, PropErr _properr = PropErr::T);

	virtual void Forward(std::vector< std::shared_ptr<Variable> >& operands, 
						 std::vector< std::shared_ptr<Variable> >& outputs, 
						 Phase phase) override;

	virtual void Backward(std::vector< std::shared_ptr<Variable> >& operands, 
						  std::vector< bool >& isConst, 
						  std::vector< std::shared_ptr<Variable> >& outputs) override;	
};

}

#endif