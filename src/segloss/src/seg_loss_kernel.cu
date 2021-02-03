#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>
#include "reduce_kernel_v3.h"
#include "logsumexp_custom.h"
#include "cuda_error_checking.h"
#include "seg_loss_kernel.h"

#define GET_BLOCK(N, T) ((N + T -1) / T)

static __device__ void atomicAddFloat(float * const address, const float value){
  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;
  do 
    {
      assumed = old;
      old = atomicCAS(address_as_i, assumed, __float_as_int(__int_as_float(assumed) + value));
    } while (assumed != old);
}

static __device__ void atomicMulFloat(float * const address, const float value){
  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;
  do 
    {
      assumed = old;
      old = atomicCAS(address_as_i, assumed, __float_as_int(__int_as_float(assumed) * value));
    } while (assumed != old);
}


__global__ void scalar_divide(float *a, float *b, float *c){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index == 0){
    c[index] = a[index] / fmax(b[index], 1e-5);
  }
}


__global__ void f1_index(float* W, float* alpha, int t, int B, int T, int S, int V, float* tmp, int N){
  // tmp: B*SV, alpha: (T+1)*B
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    int vidx = index % V;
    int sidx = (index / V) % S;
    int bidx = (index / V) / S;
    int sidx_w = sidx;
    int tidx_w = t - 1 - sidx;
    if (tidx_w < 0){
      tmp[index] = -INFINITY;
    }
    else{
      tmp[index] = alpha[tidx_w*B+bidx] + W[bidx*T*S*V+tidx_w*S*V+sidx_w*V+vidx];
    }
  }
}

__global__ void b1_index(float* W, float* beta, int t, int B, int T, int S, int V, float* tmp, int N, int* prob_sizes){
  // tmp: B*SV, beta: (T+1)*B
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    int vidx = index % V;
    int sidx = (index / V) % S;
    int bidx = (index / V) / S;
    int sidx_w = sidx;
    int tidx_w = t;
    int tidx_beta = t + sidx + 1;
    int max_t = prob_sizes[bidx];
    if (tidx_beta > max_t){
      tmp[index] = -INFINITY;
    }
    else{
      tmp[index] = beta[tidx_beta*B+bidx] + W[bidx*T*S*V+tidx_w*S*V+sidx_w*V+vidx];
    }
  }
}

__global__ void f2_index(float *W, int *label_batch, float *alpha, const int iy, const int B, const int T, const int S, const int V, const int L, float *tmp, const int N){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    int s = index % S;
    int t = (index / S) % (T+1);
    int b = (index / S) / (T+1);
    int l = label_batch[b*L+iy-1];
    // assert(index >= 0 && index < B*(T+1)*S);
    if (t-s-1 < 0 || l == -1){
      tmp[index] = -INFINITY;
    }
    else{
      // assert((iy-1)*B*(T+1)+b*(T+1)+(t-s-1) >= 0 && (iy-1)*B*(T+1)+b*(T+1)+(t-s-1) < (T+1)*B*(L+1));
      // assert(b*T*S*V+(t-s-1)*S*V+s*V+l >= 0 && b*T*S*V+(t-s-1)*S*V+s*V+l <B*T*S*V);
      tmp[index] = alpha[(iy-1)*B*(T+1)+b*(T+1)+(t-s-1)] + W[b*T*S*V+(t-s-1)*S*V+s*V+l];
    }
  }
}

__global__ void b2_index(float *W, int *label_batch, float *beta, const int iy, const int B, const int T, const int S, const int V, const int L, float *tmp, const int N){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < N){
    int s = index % S;
    int t = (index / S) % (T+1);
    int b = (index / S) / (T+1);
    int l = label_batch[b*L+iy];
    if (t+s+1 > T || l == -1){
      tmp[index] = -INFINITY;
    }
    else{
      tmp[index] = beta[(iy+1)*B*(T+1)+b*(T+1)+(t+s+1)] + W[b*T*S*V+t*S*V+s*V+l];
    }
  }
}

__global__ void b1_init(float* beta, int* prob_sizes, int Ty, int Bx){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < Ty*Bx){
    int b = index % Bx;
    int t = index / Bx;
    if (t == prob_sizes[b]){
      beta[index] = 0;
    }
    else{
      beta[index] = -INFINITY;
    }
  }
}

__global__ void b1_assign(float* dst, float* src, int t, int* prob_sizes, int B){
  int b = threadIdx.x + blockDim.x * blockIdx.x;
  if (b < B){
    if (t != prob_sizes[b]){
      dst[b] = src[b];
    }
  }
}

__global__ void f2_init(float *alpha, const int B, const int T){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int N = B*(T+1);
  if (index < N){
    int b = index / (T+1);
    int t = index % (T+1);
    if (t == 0){
      alpha[index] = 0;
    }
    else{
      alpha[index] = -INFINITY;
    }
  }
}

__global__ void b2_init(float *beta, int *prob_sizes, int *label_sizes, const int B, const int T, const int L){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int N = (L+1)*B*(T+1);
  if (index < N){
    int t = index % (T + 1);
    int b = (index / (T + 1)) % B;
    int iy = index / ((T + 1)* B);
    if (iy == label_sizes[b] && t == prob_sizes[b]){
      beta[index] = 0;
    }
    else{
      beta[index] = -INFINITY;
    }
  }
}

__global__ void b2_assign(float *beta, float *res, int *prob_sizes, int *label_sizes, const int B, const int T, const int iy){
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int N = B*(T+1);
  if (index < N){
    int t = index % (T+1);
    int b = index / (T+1);
    if ( t!= prob_sizes[b] && iy != label_sizes[b]){
      beta[index] = res[index];
    }
  }
}


__global__ void loss_comp(float *alpha_1d, float *alpha_2d, int *prob_sizes, int *label_sizes, float *loss, float *valid_batch, const int B, const int T, const int S, bool is_sum){
  /* alpha_1d: (T+1)*B array, alpha_2d: (L+1)*B*(T+1) array,  valid_batch: B array */

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int index_1d, index_2d;
  float single_loss;
  if (index < B){
    if (prob_sizes[index] >= label_sizes[index] && prob_sizes[index] <= S * label_sizes[index]){
      index_1d = prob_sizes[index] * B + index;
      index_2d = label_sizes[index] * B * (T+1) + index * (T+1) + prob_sizes[index];
      single_loss = alpha_1d[index_1d] - alpha_2d[index_2d];
      if (!is_sum){
        single_loss /= (float) prob_sizes[index];
      }
      if (isnan(single_loss) == 0){
        loss[index] = single_loss;
        valid_batch[index] = 1;
      }
      else{
        loss[index] = 0;
        valid_batch[index] = 0;
      }
    }
    else{
      loss[index] = 0;
      valid_batch[index] = 0;
    }
  }
}

__global__ void get_valid_batch(int *prob_sizes, int *label_sizes, float *batch_mask, const int B, const int S){
  int b = threadIdx.x + blockIdx.x * blockDim.x;
  if (b < B){
    if (prob_sizes[b] >= label_sizes[b] && prob_sizes[b] <= S * label_sizes[b]){
      batch_mask[b] = 1;
    }
    else{
      batch_mask[b] = 0;
    }
  }
}

__global__ void grad_comp_1d(float* W, float *grad_W, int *prob_sizes, float *alpha, float* beta, float *batch_mask, const int B, const int T, const int S, const int V, const int N){
  // int N = B*T*S*V;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < N){
    int v = index % V;
    int s = (index / V) % S;
    int t = (index / (V*S)) % T;
    int b = index / (V*S*T);
    float log_z = beta[b];
    float alpha_val = t < prob_sizes[b]+1?alpha[t*B+b]:-INFINITY;
    float beta_val = (t+s+1) < prob_sizes[b]+1?beta[(t+s+1)*B+b]:-INFINITY;
    grad_W[index] = (batch_mask[b] == 1)?expf(alpha_val + W[index] + beta_val - log_z):0;
    index += gridDim.x * blockDim.x;
  }
}

__global__ void grad_comp_2d(float *W, float *grad_W, int *prob_sizes, int *label_sizes, int *label_batch, float *alpha, float *beta, float* batch_mask, const int B, const int T, const int S, const int V, const int L, const int N){
  // int N = B*T*S*L;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < N){
    int iy = index % L;
    int s = (index / L) % S;
    int t = (index / (S*L)) % T;
    int b = index / (T*S*L);
    float log_z = beta[b*(T+1)];
    int v = label_batch[b*L+iy];
    float alpha_val = (iy<label_sizes[b]+1 && t<prob_sizes[b]+1)?alpha[iy*B*(T+1)+b*(T+1)+t]:-INFINITY;
    float beta_val = (iy+1<label_sizes[b]+1 && t+s+1<prob_sizes[b]+1)?beta[(iy+1)*B*(T+1)+b*(T+1)+t+s+1]:-INFINITY;
    int w_index = b*T*S*V+t*S*V+s*V+v;
    if (v >= 0){
      assert(w_index < B*T*S*V && w_index >=0);
      // if (w_index >= B*T*S*V || w_index < 0){
      //   assert(w_index < B*T*S*V && w_index >=0);
      // }
      float val_to_add = (batch_mask[b] == 1)?expf(alpha_val + W[w_index] + beta_val - log_z):0;
      atomicAddFloat(grad_W+w_index, -val_to_add);
    }

    index += gridDim.x * blockDim.x;
  }
}

__global__ void scale_grad(float *grad_W, float grad_output, float *batch_mask, int *prob_sizes, const bool is_sum, const float* num_valid_batch, const int num_cols, const int N){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < N){
    int b = index / num_cols;
    float coef;
    if (is_sum){
      coef = grad_output / fmax(num_valid_batch[0], 1);
    }
    else{
      coef = grad_output / fmax(num_valid_batch[0] * prob_sizes[b], 1);
    }
    if (isnan(grad_W[index]) != 0){
      grad_W[index] = 0;
    }
    else{
      atomicMulFloat(grad_W+index, coef);
    }

    index += gridDim.x * blockDim.x;
  }
}

void forward_1d(float* W, float* alpha, float* buffer_1d, float* tmp_1, float* tmp_2, int B, int T, int S, int V, const int nTB_reduce, const int nBLK_reduce){
  // alpha: (T+1)*B array, buffer_1d: B*(S*V) array, tmp_1: B*nBLK_reduce array, tmp_2: B array
  int num_threads = 256;
  int N = B*S*V;
  int num_blocks = GET_BLOCK(N, num_threads);
  // W: [B, T, S, V], alpha: [B, T+1], tmp: >= [B, S*V]
  cudaMemset(alpha, 0, B*sizeof(float));
  for(int t=1; t < T+1; t++){
    // broadcast vector addition
    f1_index<<<num_blocks, num_threads>>>(W, alpha, t, B, T, S, V, buffer_1d, N);
    // checkLastCUDAError("f1_index");
    logsumexp_wide(buffer_1d, tmp_1, tmp_2, alpha+t*B, B, S*V, nTB_reduce, nBLK_reduce);
  }
}

void backward_1d(float* W, float* beta, float* buffer_1d, float* tmp_1, float* tmp_2, float* tmp_3, int B, int T, int S, int V, const int nTB_reduce, const int nBLK_reduce, int* prob_sizes){
  // beta: (T+1)*B array, buffer_1d: B*(S*V) array, tmp_1: B*nBLK_reduce array, tmp_2: B array, tmp_3: B array
  int num_threads = 256;
  int N = B*S*V;
  int num_blocks = GET_BLOCK(N, num_threads);
  // W: [B, T, S, V], alpha: [B, T+1], tmp: >= [B, S*V]

  // Initialize beta
  int num_blocks_init = GET_BLOCK((T+1)*B, num_threads);
  b1_init<<<num_blocks_init, num_threads>>>(beta, prob_sizes, T+1, B);

  int num_threads_assign = 128;
  int num_blocks_assign = GET_BLOCK(B, num_threads_assign);
  for(int t=T-1; t >= 0; t--){
    b1_index<<<num_blocks, num_threads>>>(W, beta, t, B, T, S, V, buffer_1d, N, prob_sizes);
    logsumexp_wide(buffer_1d, tmp_1, tmp_2, tmp_3, B, S*V, nTB_reduce, nBLK_reduce);
    b1_assign<<<num_blocks_assign, num_threads_assign>>>(beta+t*B, tmp_3, t, prob_sizes, B);
  }
}

void forward_2d(float *W, int *label_batch, float *alpha, float *tmp_1, float *tmp_2, const int B, const int T, const int S, const int V, const int L){
  /* alpha: (L+1)*B*(T+1) array, tmp_1: B*(T+1)*S array, tmp_2: B*(T+1) array */
  /* alpha[iy, b, t] = logsumexp_s{alpha[iy-1, b, t-s-1]+W[b, t-s-1, s, L[b, iy-1]]}*/
  
  // Initialize alpha, alpha[0, :, 0] = 0, alpha[0, :, 1:] = -inf
  int N = B*(T+1)*S;
  int num_rows = B*(T+1);
  int num_threads_index = 256;
  int num_blocks_index = GET_BLOCK(N, num_threads_index);
  int num_threads_init = 256;
  int num_blocks_init = GET_BLOCK(num_rows, num_threads_init);
  f2_init<<<num_blocks_init, num_threads_init>>>(alpha, B, T);
  
  for(int iy=1; iy < L+1; iy++){
    f2_index<<<num_blocks_index, num_threads_index>>>(W, label_batch, alpha, iy, B, T, S, V, L, tmp_1, N);
    logsumexp_narrow(tmp_1, tmp_2, alpha+iy*B*(T+1), B*(T+1), S);
  }
}


void backward_2d(float *W, int *label_batch, int *prob_sizes, int *label_sizes, float *beta, float *tmp_1, float *tmp_2, float *tmp_3, const int B, const int T, const int S, const int V, const int L){
  /* beta: (L+1)*B*(T+1) array, tmp_1: B*(T+1)*S array, tmp_2: B*(T+1) array, tmp_3: B*(T+1) array */
  /* beta[iy, b, t] = logsumexp_s{beta[iy+1, b, t+s+1]+W[b, t, s, L[b, iy]]} */

  // Initialize beta, beta[-1, :, -1] = 0, beta[-1:, :, -1:] = -inf, -1: prob_size, label_size
  int N = B*(T+1)*S;
  int num_rows = B*(T+1);
  int num_threads_index = 256;
  int num_blocks_index = GET_BLOCK(N, num_threads_index);
  int num_threads_assign = 256;
  int num_blocks_assign = GET_BLOCK(num_rows, num_threads_assign);
  int num_threads_init = 256;
  int num_blocks_init = GET_BLOCK(B*(T+1)*(L+1), num_threads_init);
  b2_init<<<num_blocks_init, num_threads_init>>>(beta, prob_sizes, label_sizes, B, T, L);
  for(int iy=L-1; iy >= 0; iy--){
    b2_index<<<num_blocks_index, num_threads_index>>>(W, label_batch, beta, iy, B, T, S, V, L, tmp_1, N);
    logsumexp_narrow(tmp_1, tmp_2, tmp_3, num_rows, S);
    b2_assign<<<num_blocks_assign, num_threads_assign>>>(beta+iy*num_rows, tmp_3, prob_sizes, label_sizes, B, T, iy);
  }
}


void SegLoss::seg_forward(float *W, float *loss, int *label_sizes, int *prob_sizes, int *label_batch, float *alpha_1d, float *alpha_2d, float *beta_1d, float *beta_2d, const int B, const int T, const int S, const int V, const int L, const bool is_sum){

  int nBLK_reduce = T2B_REDUCE(chunk_size, S*V);

  check_cache(buffer_1d, B*S*V);
  check_cache(buffer_2d, B*(T+1)*S);
  check_cache(tmp_1d_1, B*nBLK_reduce);
  check_cache(tmp_1d_2, B);
  check_cache(tmp_1d_3, B);
  check_cache(tmp_2d_1, B*(T+1));
  check_cache(tmp_2d_2, B*(T+1));

  forward_1d(W, alpha_1d, buffer_1d->data, tmp_1d_1->data, tmp_1d_2->data, B, T, S, V, chunk_size, nBLK_reduce);
  forward_2d(W, label_batch, alpha_2d, buffer_2d->data, tmp_2d_1->data, B, T, S, V, L);

  backward_1d(W, beta_1d, buffer_1d->data, tmp_1d_1->data, tmp_1d_2->data, tmp_1d_3->data, B, T, S, V, chunk_size, nBLK_reduce, prob_sizes);
  backward_2d(W, label_batch, prob_sizes, label_sizes, beta_2d, buffer_2d->data, tmp_2d_1->data, tmp_2d_2->data, B, T, S, V, L);
  
  int num_threads = 128;
  int num_blocks = GET_BLOCK(B, num_threads);
  // Using first B values of tmp_1d_blk and tmp_1d_max to store loss and if valid batch
  // bool is_sum = (reduction == LossReduceOption::SUM)?true:false;
  // cudaDeviceSynchronize();

  loss_comp<<<num_blocks, num_threads>>>(alpha_1d, alpha_2d, prob_sizes, label_sizes, tmp_1d_2->data, tmp_1d_3->data, B, T, S, is_sum);
  // checkLastCUDAError("loss_comp");
  device_reduce_narrow(tmp_1d_2->data, &(tmp_1d_2->data[0]), 1, B, CudaReduceOption::SUM);
  device_reduce_narrow(tmp_1d_3->data, &(tmp_1d_3->data[0]), 1, B, CudaReduceOption::SUM);
  scalar_divide<<<1, 1>>>(tmp_1d_2->data, tmp_1d_3->data, loss);
}

void SegLoss::seg_backward(float *W, float *grad_W, float grad_output, int *label_sizes, int *prob_sizes, int *label_batch, float *alpha_1d, float *alpha_2d, float *beta_1d, float *beta_2d, const int B, const int T, const int S, const int V, const int L, const bool is_sum){

  int N1, N2;
  N1 = B*T*S*V;
  N2 = B*T*S*L;
  int num_threads_mask = 128;
  int num_block_mask = GET_BLOCK(B, num_threads_mask);
  // checkCUDAError(cudaSetDevice(device_id));

  float *batch_mask, *num_valid_batch;
  checkCUDAError(cudaMalloc(&batch_mask, B * sizeof(float)));
  checkCUDAError(cudaMalloc(&num_valid_batch, 1 * sizeof(float)));

  // valid_batch: [B], store if batch valid or not
  get_valid_batch<<<num_block_mask, num_threads_mask>>>(prob_sizes, label_sizes, batch_mask, B, S);
  device_reduce_narrow(batch_mask, num_valid_batch, 1, B, CudaReduceOption::SUM);

  int num_threads_1d = 256;
  int num_threads_2d = 256;
  int num_block_1d = GET_BLOCK(N1, num_threads_1d);
  int num_block_2d = GET_BLOCK(N2, num_threads_2d);
  grad_comp_1d<<<num_block_1d, num_threads_1d>>>(W, grad_W, prob_sizes, alpha_1d, beta_1d, batch_mask, B, T, S, V, N1);

  grad_comp_2d<<<num_block_2d, num_threads_2d>>>(W, grad_W, prob_sizes, label_sizes, label_batch, alpha_2d, beta_2d, batch_mask, B, T, S, V, L, N2);
  scale_grad<<<num_block_1d, num_threads_1d>>>(grad_W, grad_output, batch_mask, prob_sizes, is_sum, num_valid_batch, T*S*V, N1);
  checkCUDAError(cudaFree(batch_mask));
  checkCUDAError(cudaFree(num_valid_batch));

}

Cache* SegLoss::init_cache(){
  Cache* ptr = (Cache*) malloc(sizeof(Cache));
  ptr->size = 0;
  ptr->data = NULL;
  return ptr;
}

void SegLoss::check_cache(Cache* ptr, int new_num){
  int old_num = ptr->size;
  if (old_num < new_num){
    if (verbose){
      printf("\nResizing cache from %d into %d\n", old_num, new_num);
    }
    ptr->size = new_num;
    checkCUDAError(cudaFree(ptr->data));
    checkCUDAError(cudaMalloc(&(ptr->data), new_num*sizeof(float)));
  }
}

void SegLoss::free_cache(Cache* ptr){
  checkCUDAError(cudaFree(ptr->data));
  free(ptr);
}

SegLoss::SegLoss(int chunk_size, const bool verbose) : verbose(verbose), chunk_size(chunk_size){
  buffer_1d = init_cache();
  buffer_2d = init_cache();
  tmp_1d_1 = init_cache();
  tmp_1d_2 = init_cache();
  tmp_1d_3 = init_cache();
  tmp_2d_1 = init_cache();
  tmp_2d_2 = init_cache();
}

SegLoss::~SegLoss(){

  if (verbose){
    printf("\nFreeing buffers \n");
  }
  free_cache(buffer_1d);
  free_cache(buffer_2d);
  free_cache(tmp_1d_1);
  free_cache(tmp_1d_2);
  free_cache(tmp_1d_3);
  free_cache(tmp_2d_1);
  free_cache(tmp_2d_2);
}
