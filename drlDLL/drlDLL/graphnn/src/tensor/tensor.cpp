#include "tensor/tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_dense_tensor.h"
#include <cassert>

#ifndef USE_GPU
#define USE_GPU
#endif

namespace gnn
{

template<typename mode, typename matType, typename Dtype>
TensorTemplate<mode, matType, Dtype>& Tensor::Derived()
{
	return *(dynamic_cast<TensorTemplate<mode, matType, Dtype>*>(this));
}

void Tensor::Serialize(FILE* fid)
{
	size_t len = shape.dims.size();
	fwrite(&len, sizeof(size_t), 1, fid);
	fwrite(shape.dims.data(), sizeof(size_t), shape.dims.size(), fid);
	//assert(fwrite(&len, sizeof(size_t), 1, fid) == 1);
	//assert(fwrite(shape.dims.data(), sizeof(size_t), shape.dims.size(), fid) == shape.dims.size());
}

void Tensor::Deserialize(FILE* fid)
{
	size_t len;
	if (fread(&len, sizeof(size_t), 1, fid) != 1)
		std::cout << "Error in reading size [Tensor::Deserialize]";
	//assert(fread(&len, sizeof(size_t), 1, fid) == 1);
	std::vector<size_t> dims(len);
	if (fread(dims.data(), sizeof(size_t), len, fid)!=len)
		std::cout << "Error in reading data [Tensor::Deserialize]";
	//assert(fread(dims.data(), sizeof(size_t), len, fid) == len);	
	shape.Reshape(dims);
}

template TensorTemplate<CPU, DENSE, float>& Tensor::Derived<CPU, DENSE, float>(); 
template TensorTemplate<CPU, DENSE, double>& Tensor::Derived<CPU, DENSE, double>(); 
template TensorTemplate<CPU, DENSE, int>& Tensor::Derived<CPU, DENSE, int>(); 
#ifdef USE_GPU
template TensorTemplate<GPU, DENSE, float>& Tensor::Derived<GPU, DENSE, float>(); 
template TensorTemplate<GPU, DENSE, double>& Tensor::Derived<GPU, DENSE, double>(); 
template TensorTemplate<GPU, DENSE, int>& Tensor::Derived<GPU, DENSE, int>(); 
#endif

}