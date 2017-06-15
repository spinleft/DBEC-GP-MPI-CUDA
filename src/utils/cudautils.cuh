/**
 * DBEC-GP-OMP-CUDA-MPI programs are developed by:
 *
 * Vladimir Loncar, Antun Balaz
 * (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
 *
 * Srdjan Skrbic
 * (Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia)
 *
 * Paulsamy Muruganandam
 * (Bharathidasan University, Tamil Nadu, India)
 *
 * Luis E. Young-S, Sadhan K. Adhikari
 * (UNESP - Sao Paulo State University, Brazil)
 *
 *
 * Public use and modification of these codes are allowed provided that the
 * following papers are cited:
 * [1] V. Loncar et al., Comput. Phys. Commun. 209 (2016) 190.      
 * [2] V. Loncar et al., Comput. Phys. Commun. 200 (2016) 406.      
 * [3] R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.
 *
 * The authors would be grateful for all information and/or comments
 * regarding the use of the programs.
 */

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

#define CUDA_BLOCK_SIZE     128
#define CUDA_BLOCK_SIZE_2D  16
#define CUDA_BLOCK_SIZE_3D  8

#define TID_X (blockIdx.x * blockDim.x + threadIdx.x)
#define TID_Y (blockIdx.y * blockDim.y + threadIdx.y)
#define TID_Z (blockIdx.z * blockDim.z + threadIdx.z)

#define GRID_STRIDE_X (blockDim.x * gridDim.x)
#define GRID_STRIDE_Y (blockDim.y * gridDim.y)
#define GRID_STRIDE_Z (blockDim.z * gridDim.z)

#define cudaCheckError(ans) { cudaCheck((ans), __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static const char *cudaGetFFTErrorString(cufftResult error) {
   switch (error) {
   case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

   case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

   case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

   case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

   case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

   case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

   case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

   case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

   case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

   case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
   }

   return "<unknown>";
}

#define cudaCheckFFTError(ans) { cudaFFTCheck((ans), __FILE__, __LINE__); }
inline void cudaFFTCheck(cufftResult code, const char *file, int line, bool abort=true) {
   if (code != CUFFT_SUCCESS) {
      fprintf(stderr,"cuFFT error: %s %s %d\n", cudaGetFFTErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ __host__ __inline__ double *get_double_matrix_row(cudaPitchedPtr cudaptr, long row) {
   return (double *)((char *)cudaptr.ptr + row * cudaptr.pitch);
}

__device__ __host__ __inline__ cuDoubleComplex *get_complex_matrix_row(cudaPitchedPtr cudaptr, long row) {
   return (cuDoubleComplex *)((char *)cudaptr.ptr + row * cudaptr.pitch);
}

__device__ __host__ __inline__ double *get_double_tensor_row(cudaPitchedPtr cudaptr, long slice, long row) {
   return (double *)((((char *)cudaptr.ptr) + slice * cudaptr.pitch * cudaptr.ysize) + row * cudaptr.pitch);
}

__device__ __host__ __inline__ cuDoubleComplex *get_complex_tensor_row(cudaPitchedPtr cudaptr, long slice, long row) {
   return (cuDoubleComplex *)((((char *)cudaptr.ptr) + slice * cudaptr.pitch * cudaptr.ysize) + row * cudaptr.pitch);
}

__device__ __inline__ cuDoubleComplex cuCexp(double arg) {
   cuDoubleComplex res;

   sincos(arg, &res.y, &res.x);

   return res;
}

#endif /* CUDAUTILS_H */
