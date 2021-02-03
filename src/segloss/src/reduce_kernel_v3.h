#ifndef REDUCE_KERNEL_H
#define REDUCE_KERNEL_H

const int block_x = 32;
const int block_y = 32;
const int horizontal_stride = block_x;

enum class CudaReduceOption{SUM, MAX};

void device_reduce(float *data_in, float *buffer, float *data_out, int num_rows, int num_cols, int num_threads, int num_blocks, CudaReduceOption opt);
  
void device_reduce_narrow(float *data_in, float *data_out, int num_rows, int num_cols, CudaReduceOption opt);

#endif
