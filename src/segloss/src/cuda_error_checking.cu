// cuda_error_checking.h
#include "cuda_error_checking.h"
#include <stdio.h>
 
// Assumes single device when calling cudaDeviceReset(); and exit(code);
// In some cases a more lengthy program clean up / termination may be needed
 
void checkError(cudaError_t code, char const * func, const char *file, const int line, bool abort)
{
    if (code != cudaSuccess) 
    {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n", func, file, line, code, errorMessage);
        if (abort){
            cudaDeviceReset();
            exit(code);
        }
    }
    else if (PRINT_ON_SUCCESS)
    {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n", func, file, line, code, errorMessage);
    }
}
 
void checkLastError(char const * func, const char *file, const int line, bool abort)
{
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n", func, file, line, code, errorMessage);
        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }
    else if (PRINT_ON_SUCCESS)
    {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n", func, file, line, code, errorMessage);
    }
}
