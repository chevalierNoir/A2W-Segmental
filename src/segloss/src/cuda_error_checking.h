// cuda_error_checking.h
#ifndef CHECK_CUDA_ERROR_M_H
#define CHECK_CUDA_ERROR_M_H
 
#define PRINT_ON_SUCCESS 0
 
// To be used around calls that return an error code, ex. cudaDeviceSynchronize or cudaMallocManaged
void checkError(cudaError_t code, char const * func, const char *file, const int line, bool abort = true);
#define checkCUDAError(val) { checkError((val), #val, __FILE__, __LINE__); }    // in-line regular function
#define checkCUDAError2(val) check((val), #val, __FILE__, __LINE__) // typical macro 
 
// To be used after calls that do not return an error code, ex. kernels to check kernel launch errors
void checkLastError(char const * func, const char *file, const int line, bool abort = true);
#define checkLastCUDAError(func) { checkLastError(func, __FILE__, __LINE__); }
#define checkLastCUDAError_noAbort(func) { checkLastError(func, __FILE__, __LINE__, 0); }
 
#endif // CHECK_CUDA_ERROR_M_H
