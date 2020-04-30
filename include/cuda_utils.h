#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cfloat>
#include <cassert>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

//===
static void hdlCudaErr(cudaError err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define CUDA_ALLOC_DEV_MEM(devPtr,size) hdlCudaErr(cudaMalloc(devPtr,size), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_MEMCPY_H2D(dst,src,count) hdlCudaErr(cudaMemcpy(dst,src,count,cudaMemcpyHostToDevice), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_MEMCPY_D2H(dst,src,count) hdlCudaErr(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToHost), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_MEMCPY_D2D(dst,src,count) hdlCudaErr(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToDevice), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_FREE_DEV_MEM(devPtr) hdlCudaErr(cudaFree(devPtr), __FUNCTION__, __FILE__, __LINE__)
#define CUDA_DEV_MEMSET(devPtr,value,count) hdlCudaErr(cudaMemset(devPtr,value,count), __FUNCTION__, __FILE__, __LINE__)

typedef thrust::device_system_tag device_memspace;
typedef thrust::host_system_tag host_memspace;


template <typename T, typename M>
struct ptr_type {};

template <typename T>
struct ptr_type<T, host_memspace> {
  typedef T* type;
};
template <typename T>
struct ptr_type<T, device_memspace> {
  typedef thrust::device_ptr<T> type;
};
template <typename T, typename M>
struct vector_type {};

template <typename T>
struct vector_type<T, host_memspace> {
  typedef thrust::host_vector<T> type;
};
template <typename T>
struct vector_type<T, device_memspace> {
  typedef thrust::device_vector<T> type;
};


__global__ void adjustInc(int *d_out,int* d_inc,int numElems)
{
	int id=(blockIdx.x*blockDim.x+threadIdx.x)*2;
	if(id<numElems)
	{
		d_out[id]+=d_inc[blockIdx.x];
	}
	if(id+1<numElems)
	{
		d_out[id+1]+=d_inc[blockIdx.x];
	}
}

__global__ void ExclusiveScan(int  *d_out, const int* d_in, size_t input_size, int* blockSums)
{
	extern __shared__ int data[];
	int tid = threadIdx.x;
	int offset = 1;
	int abs_start = 2*blockDim.x*blockIdx.x;

	data[2 * tid] =(abs_start+2*tid)<input_size? d_in[abs_start+2 * tid]:0;
	data[2 * tid+1] = (abs_start + 2 * tid+1)<input_size ? d_in[abs_start+2 * tid+1]:0;

	for (int d = (2 * blockDim.x) >>1; d>0; d>>=1) {
		__syncthreads();

		if (tid < d) {
			int ai = offset*(2 * tid + 1) - 1;
			int bi = offset*(2 * tid + 2) - 1;

			data[bi] += data[ai];
		}
		offset <<= 1;
	}
	// 0 1 2 3; 0 1 2 5; 0 1 2 6
	if (tid == 0)data[2*blockDim.x - 1] = 0;

	for (int d = 1; d < 2 * blockDim.x; d<<=1) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2 * tid + 1) - 1;
			int bi = offset*(2 * tid + 2) - 1;
			int t = data[ai];
			data[ai] = data[bi];
			data[bi] += t;
		}
	}
	// 0 1 2 0; 0 0 2 1; 0 0 1 3
	__syncthreads();

	if (abs_start + 2 * tid < input_size) {
		d_out[abs_start + 2 * tid] = data[2 * tid];
	}
	if (abs_start + 2 * tid+1 < input_size) {
		d_out[abs_start + 2 * tid+1] = data[2 * tid+1];
	}

	if (tid == 0) {
		blockSums[blockIdx.x] = data[blockDim.x * 2 - 1];// 3
		if(abs_start + blockDim.x * 2 - 1<input_size)blockSums[blockIdx.x]+=d_in[abs_start + blockDim.x * 2 - 1];//3+3=6

	}

}

__device__
int serailScan(int* out,int* pred, int size)
{
	int acc=0;
	for(int i=0;i<size;i++)
	{
		out[i]=acc;
		acc=acc+pred[i];
	}
	return acc;
}
__device__
int PrefixSum(int* d_scan, int *d_pred, int numElems)
{
	int block_size=256;
	int num_double_blocks=(numElems%(2*block_size)==0)?(numElems/(2*block_size)):(numElems/(2*block_size)+1);//ceil(1.0f*numElems/(2*block_size));
	int* d_blk_offsets=(int*)malloc(sizeof(int)*num_double_blocks);
	if(d_blk_offsets==NULL)
		assert(false); // insufficient mem

	ExclusiveScan<<<num_double_blocks,block_size,2*block_size*sizeof(int)>>>
			(d_scan,d_pred,numElems,d_blk_offsets);
	// without this, parent cannot see the operation made by child
	cudaDeviceSynchronize();

	int finalSum;
	if(num_double_blocks>1)
	{
		int* d_scan_temp=(int*)malloc(sizeof(int)*num_double_blocks);
		if(d_scan_temp==NULL)
			assert(false); // insufficient mem
/// can not recurse in dynamic parallelism...
		finalSum=serailScan(d_scan_temp,d_blk_offsets,num_double_blocks);
//		finalSum=PrefixSum(d_scan_temp,d_blk_offsets,num_double_blocks);

		adjustInc<<<num_double_blocks,block_size>>>(d_scan,d_scan_temp,numElems);
		cudaDeviceSynchronize();
		free(d_scan_temp);
	}else
	{
		finalSum=d_blk_offsets[0];
	}
	free(d_blk_offsets);
	return finalSum;
}

#endif // CUDA_UTILS_H
