#ifndef SEG_LOSS_KERNEL_H
#define SEG_LOSS_KERNEL_H

typedef struct Caches{
  int size;
  float* data;
} Cache;

class SegLoss
{
 private:
  Cache *buffer_1d, *buffer_2d;
  Cache *tmp_1d_1, *tmp_1d_2, *tmp_1d_3, *tmp_2d_1, *tmp_2d_2;
  int chunk_size;
  bool verbose;
  Cache* init_cache();
  void check_cache(Cache *ptr, int new_num);
  void free_cache(Cache *ptr);

 public:
  SegLoss(int chunk_size, const bool verbose = true);
  void seg_forward(float *W, float *loss, int *label_sizes, int *prob_sizes, int *label_batch, float *alpha_1d, float *alpha_2d, float *beta_1d, float *beta_2d, const int B, const int T, const int S, const int V, const int L, const bool is_sum);
  void seg_backward(float *W, float *grad_W, float grad_output, int *label_sizes, int *prob_sizes, int *label_batch, float *alpha_1d, float *alpha_2d, float *beta_1d, float *beta_2d, const int B, const int T, const int S, const int V, const int L, const bool is_sum);
  ~SegLoss();
};

#endif
