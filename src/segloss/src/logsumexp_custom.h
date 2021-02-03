#ifndef LOGSUMEXP_CUSTOM_H
#define LOGSUMEXP_CUSTOM_H

#define T2B_REDUCE(nTB, nCol) (min((nCol + nTB - 1) / nTB, 1024))
#define BLK_SIZE(nTB, N) ((N + nTB - 1)/nTB)

void logsumexp_wide(float *data_in, float *buffer_1, float *buffer_2, float *data_out, int num_rows, int num_cols, int nTB_reduce, int nBLK_reduce);

void logsumexp_narrow(float *data_in, float *buffer, float *data_out, int num_rows, int num_cols);

#endif
