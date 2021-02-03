#include "reduce_kernel_v3.h"
#include "logsumexp_custom.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

__global__ void exp_kernel(float *data, float *max, const int num_cols, const int N){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    int rid = index / num_cols;
    int cid = index % num_cols;
    data[index] = expf(data[index] - max[rid]);
  }
}

__global__ void log_kernel(float *data, float *max, const int N){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    data[index] = logf(data[index]) + max[index];
  }
}

__global__ void revise_max_kernel(float *data, const int N){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    if (data[index] == -INFINITY){
      data[index] = 0;
    }
  }
}

__global__ void set_kernel(float *data, const float val, const int N){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    data[index] = val;
  }
}


void logsumexp_narrow(float *data_in, float *buffer, float *data_out, int num_rows, int num_cols){
  /* buffer: num_rows */

  // max(x)
  device_reduce_narrow(data_in, buffer, num_rows, num_cols, CudaReduceOption::MAX);
  int N = num_rows * num_cols;
  int nTB_sub = 512;
  int nBLK_sub = BLK_SIZE(nTB_sub, N); // (N + nTB_sub - 1) / nTB_sub;

  //\exp(x-\max(x))
  revise_max_kernel<<<BLK_SIZE(nTB_sub, num_rows), nTB_sub>>>(buffer, num_rows);
  exp_kernel<<<nBLK_sub, nTB_sub>>>(data_in, buffer, num_cols, N);

  // sum(exp(x-max(x)))
  device_reduce_narrow(data_in, data_out, num_rows, num_cols, CudaReduceOption::SUM);

  //log(sum(x-max(x))) + max(x)
  float xn = log2((float) num_rows);
  int nTB_log = min((int) pow(2, ceil(log2((float) num_rows))), 1024);
  int nBLK_log = BLK_SIZE(nTB_log, num_rows); // (num_rows + nTB_log - 1) / nTB_log;
  log_kernel<<<nBLK_log, nTB_log>>>(data_out, buffer, num_rows);
}

void logsumexp_wide(float *data_in, float *buffer_1, float *buffer_2, float *data_out, int num_rows, int num_cols, int nTB_reduce, int nBLK_reduce){
  /* data_in: num_rows*num_cols, buffer_1: num_rows * nBLK_reduce, buffer_2: num_rows, data_out: num_rows */

  assert(nBLK_reduce == T2B_REDUCE(nTB_reduce, num_cols));
  int nTB_set = 512;
  int bf1_size, bf2_size;
  bf1_size = nBLK_reduce * num_rows;
  bf2_size = num_rows;

  set_kernel<<<BLK_SIZE(nTB_set, bf1_size), nTB_set>>>(buffer_1, -INFINITY, bf1_size);
  set_kernel<<<BLK_SIZE(nTB_set, bf2_size), nTB_set>>>(buffer_2, -INFINITY, bf2_size);

  // max(x)
  device_reduce(data_in, buffer_1, buffer_2, num_rows, num_cols, nTB_reduce, nBLK_reduce, CudaReduceOption::MAX);
  revise_max_kernel<<<BLK_SIZE(nTB_set, num_rows), nTB_set>>>(buffer_2, num_rows);
  
  int N = num_rows * num_cols;
  int nTB_sub = 512;
  int nBLK_sub = (N + nTB_sub - 1) / nTB_sub;

  //exp(x-max(x))
  exp_kernel<<<nBLK_sub, nTB_sub>>>(data_in, buffer_2, num_cols, N);

  // sum(exp(x-max(x)))
  set_kernel<<<BLK_SIZE(nTB_set, bf1_size), nTB_set>>>(buffer_1, 0, bf1_size);
  set_kernel<<<BLK_SIZE(nTB_set, bf2_size), nTB_set>>>(data_out, 0, bf2_size);
  device_reduce(data_in, buffer_1, data_out, num_rows, num_cols, nTB_reduce, nBLK_reduce, CudaReduceOption::SUM);

  //log(sum(x-max(x))) + max(x)
  float xn = log2((float) num_rows);
  int nTB_log = min((int) pow(2, ceil(log2((float) num_rows))), 1024);
  int nBLK_log = (num_rows + nTB_log - 1) / nTB_log;
  log_kernel<<<nBLK_log, nTB_log>>>(data_out, buffer_2, num_rows);
}
