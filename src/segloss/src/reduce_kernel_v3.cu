#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include "reduce_kernel_v3.h"

__device__ void atomicMaxFloat(float * const address, const float value){
  if (* address >= value)
    {
      return;
    }
  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;
  do 
    {
      assumed = old;
      if (__int_as_float(assumed) >= value)
        {
          break;
        }
      old = atomicCAS(address_as_i, assumed, __float_as_int(value));
    } while (assumed != old);
}


static __device__ void atomicAddFloat(float * const address, const float value){
  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;
  do 
    {
      assumed = old;
      old = atomicCAS(address_as_i, assumed, __float_as_int(__int_as_float(assumed) + value));
    } while (assumed != old);
}


__global__ void block_reduce_kernel_narrow(const float *data_in, float *data_out, int width, int height, bool is_max){
  // Using width/height
  float base_val = is_max?-INFINITY:0;
  __shared__ float sdata[block_y+1][block_x];
  int idx = threadIdx.y + blockDim.y * blockIdx.y;
  int height_stride = gridDim.y * blockDim.y;
  int full_height = (height & (~((unsigned long long)(block_y-1)))) +((height & (block_y-1))?block_y:0);
  for (int h = idx; h < full_height; h += height_stride){
    sdata[threadIdx.y][threadIdx.x] = base_val; // -INFINITY
    int in_ptr = h * width + threadIdx.x;;
    for (int w = threadIdx.x; w < width; w += horizontal_stride){
      if (is_max){
        sdata[threadIdx.y][threadIdx.x] = fmax(sdata[threadIdx.y][threadIdx.x], (h < height)?data_in[in_ptr]:base_val); // fmax(sdata[threadIdx.y][threadIdx.x], (h < height)?data_in[in_ptr]:-INFINITY)
      }
      else{
        sdata[threadIdx.y][threadIdx.x] += (h < height)?data_in[in_ptr]:base_val;
      }
      in_ptr += horizontal_stride;
    }
    __syncthreads();
    float my_val = sdata[threadIdx.y][threadIdx.x];
    for (int i = warpSize>>1; i > 0; i>>=1){
      if (is_max){
        my_val = fmax(my_val, __shfl_xor_sync(0xFFFFFFFU, my_val, i)); // 0xFFFFFFFU
      }
      else{
        my_val += __shfl_xor_sync(0xFFFFFFFU, my_val, i); // 0xFFFFFFFU
      }
    }
    __syncthreads();
    if (threadIdx.x == 0){
      sdata[threadIdx.y][0] = my_val;
    }
    __syncthreads();
    if ((threadIdx.x == 0) && (h < height)){
      data_out[h] = sdata[threadIdx.y][0];
    }
  }

}

__global__ void block_reduce_kernel(const float *data_in, float* data_out, int num_col_in, const int num_col_out, const int max_size, bool is_max){
  float base_val = is_max?-INFINITY:0;
  // __shared__ float shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  int idx, idy;
  float my_val = base_val;
  idx = threadIdx.x + blockDim.x * blockIdx.x;
  idy = blockIdx.y;
  while (idx < num_col_in){
    float temp = data_in[idy*num_col_in + idx];
    // my_val = fmax(temp, my_val);
    if (is_max){
      my_val = fmax(temp, my_val);
    }
    else{
      my_val += temp;
    }
    idx += blockDim.x * gridDim.x;
  }
  // my_val = warp_reduce_max(my_val);

  for (int offset = warpSize/2; offset > 0; offset /= 2){
    if (is_max){
      my_val = fmax(my_val, __shfl_down_sync(0xFFFFFFF, my_val, offset));
    }
    else{
      my_val += __shfl_down_sync(0xFFFFFFF, my_val, offset);
    }
  }

  if ((threadIdx.x & (warpSize - 1)) == 0){
    int stride = blockIdx.y * num_col_out + blockIdx.x;
    assert(stride >=0 && stride < max_size);
    // all ids satisfying: threadIdx.x % warpSize == 0, add to first out[blockidy][blockidx]
    if (is_max){
      atomicMaxFloat(data_out+stride, my_val);
    }
    else{
      atomicAddFloat(data_out+stride, my_val);
    }
  }
}

void device_reduce(float *data_in, float *buffer, float *data_out, int num_rows, int num_cols, int num_threads, int num_blocks, CudaReduceOption opt){
  dim3 grids1(num_blocks, num_rows);
  dim3 threads1(num_threads, 1);
  dim3 grids2(1, num_rows);
  dim3 threads2(1024);
  bool is_max = (opt == CudaReduceOption::MAX)?true:false;
  block_reduce_kernel<<<grids1, threads1>>>(data_in, buffer, num_cols, num_blocks, num_blocks*num_rows, is_max);
  block_reduce_kernel<<<grids2, threads2>>>(buffer, data_out, num_blocks, 1, num_rows, is_max);
}

void device_reduce_narrow(float *data_in, float *data_out, int num_rows, int num_cols, CudaReduceOption opt){
  dim3 grids(1, (num_rows+block_y-1)/block_y);
  dim3 threads(block_x,block_y);
  bool is_max = (opt == CudaReduceOption::MAX)?true:false;
  block_reduce_kernel_narrow<<<grids, threads>>>(data_in, data_out, num_cols, num_rows, is_max);
}
