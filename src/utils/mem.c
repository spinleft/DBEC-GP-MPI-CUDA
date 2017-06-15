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

#include "mem.h"

/* Host memory allocation */

/**
 *    Double vector allocation
 */
double *alloc_double_vector(long Nx) {
   double *vector;

   if((vector = (double *) malloc((size_t) (Nx * sizeof(double)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the vector.\n");
      exit(EXIT_FAILURE);
   }

   return vector;
}

/**
 *    Complex vector allocation
 */
cuDoubleComplex *alloc_complex_vector(long Nx) {
   cuDoubleComplex *vector;

   if((vector = (cuDoubleComplex *) malloc((size_t) (Nx * sizeof(cuDoubleComplex)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the vector.\n");
      exit(EXIT_FAILURE);
   }

   return vector;
}

/**
 *    Double matrix allocation
 */
double **alloc_double_matrix(long Nx, long Ny) {
   long cnti;
   double **matrix;

   if((matrix = (double **) malloc((size_t) (Nx * sizeof(double *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);
   }
   if((matrix[0] = (double *) malloc((size_t) (Nx * Ny * sizeof(double)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);
   }
   for(cnti = 1; cnti < Nx; cnti ++)
      matrix[cnti] = matrix[cnti - 1] + Ny;

   return matrix;
}

/**
 *    Complex matrix allocation
 */
cuDoubleComplex **alloc_complex_matrix(long Nx, long Ny) {
   long cnti;
   cuDoubleComplex **matrix;

   if((matrix = (cuDoubleComplex **) malloc((size_t) (Nx * sizeof(cuDoubleComplex *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);
   }
   if((matrix[0] = (cuDoubleComplex *) malloc((size_t) (Nx * Ny * sizeof(cuDoubleComplex)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);
   }
   for(cnti = 1; cnti < Nx; cnti ++)
      matrix[cnti] = matrix[cnti - 1] + Ny;

   return matrix;
}

/**
 *    Double tensor allocation
 */
double ***alloc_double_tensor(long Nx, long Ny, long Nz) {
   long cnti, cntj;
   double ***tensor;

   if((tensor = (double ***) malloc((size_t) (Nx * sizeof(double **)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0] = (double **) malloc((size_t) (Nx * Ny * sizeof(double *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0][0] = (double *) malloc((size_t) (Nx * Ny * Nz * sizeof(double)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   for(cntj = 1; cntj < Ny; cntj ++)
      tensor[0][cntj] = tensor[0][cntj-1] + Nz;
   for(cnti = 1; cnti < Nx; cnti ++) {
      tensor[cnti] = tensor[cnti - 1] + Ny;
      tensor[cnti][0] = tensor[cnti - 1][0] + Ny * Nz;
      for(cntj = 1; cntj < Ny; cntj ++)
         tensor[cnti][cntj] = tensor[cnti][cntj - 1] + Nz;
   }

   return tensor;
}

/**
 *    Complex tensor allocation
 */
cuDoubleComplex ***alloc_complex_tensor(long Nx, long Ny, long Nz) {
   long cnti, cntj;
   cuDoubleComplex ***tensor;

   if((tensor = (cuDoubleComplex ***) malloc((size_t) (Nx * sizeof(cuDoubleComplex **)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0] = (cuDoubleComplex **) malloc((size_t) (Nx * Ny * sizeof(cuDoubleComplex *)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   if((tensor[0][0] = (cuDoubleComplex *) malloc((size_t) (Nx * Ny * Nz * sizeof(cuDoubleComplex)))) == NULL) {
      fprintf(stderr, "Failed to allocate memory for the tensor.\n");
      exit(EXIT_FAILURE);
   }
   for(cntj = 1; cntj < Ny; cntj ++)
      tensor[0][cntj] = tensor[0][cntj-1] + Nz;
   for(cnti = 1; cnti < Nx; cnti ++) {
      tensor[cnti] = tensor[cnti - 1] + Ny;
      tensor[cnti][0] = tensor[cnti - 1][0] + Ny * Nz;
      for(cntj = 1; cntj < Ny; cntj ++)
         tensor[cnti][cntj] = tensor[cnti][cntj - 1] + Nz;
   }

   return tensor;
}

/* Host memory release */

/**
 *    Free double vector
 */
void free_double_vector(double *vector) {
   free((char *) vector);
}

/**
 *    Free complex vector
 */
void free_complex_vector(cuDoubleComplex *vector) {
   free((char *) vector);
}

/**
 *    Free double matrix
 */
void free_double_matrix(double **matrix) {
   free((char *) matrix[0]);
   free((char *) matrix);
}

/**
 *    Free complex matrix
 */
void free_complex_matrix(cuDoubleComplex **matrix) {
   free((char *) matrix[0]);
   free((char *) matrix);
}

/**
 *    Free double tensor
 */
void free_double_tensor(double ***tensor) {
   free((char *) tensor[0][0]);
   free((char *) tensor[0]);
   free((char *) tensor);
}

/**
 *    Free complex tensor
 */
void free_complex_tensor(cuDoubleComplex ***tensor) {
   free((char *) tensor[0][0]);
   free((char *) tensor[0]);
   free((char *) tensor);
}

/* Pinned memory functions */

/**
 *    Pin allocated double matrix
 */
void pin_double_matrix(double **matrix, long Nx, long Ny) {
   if(cudaHostRegister((void *) matrix[0], Nx * Ny * sizeof(double), cudaHostRegisterMapped) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated complex matrix
 */
void pin_complex_matrix(cuDoubleComplex **matrix, long Nx, long Ny) {
   if(cudaHostRegister((void *) matrix[0], Nx * Ny * sizeof(cuDoubleComplex), cudaHostRegisterMapped) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated double tensor
 */
void pin_double_tensor(double ***tensor, long Nx, long Ny, long Nz) {
   if(cudaHostRegister((void *) tensor[0][0], Nx * Ny * Nz * sizeof(double), cudaHostRegisterMapped) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated complex tensor
 */
void pin_complex_tensor(cuDoubleComplex ***tensor, long Nx, long Ny, long Nz) {
   if(cudaHostRegister((void *) tensor[0][0], Nx * Ny * Nz * sizeof(cuDoubleComplex), cudaHostRegisterMapped) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/* Mapped memory functions */

/**
 *    Map pinned double matrix
 */
struct cudaPitchedPtr map_double_matrix(double **tensor, long Nx, long Ny) {
   double *d_tensor;

   if(cudaHostGetDevicePointer((void **) &d_tensor, tensor[0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for double matrix in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_tensor, Ny * sizeof(double), Ny * sizeof(double), Nx);
}

/**
 *    Map pinned complex matrix
 */
struct cudaPitchedPtr map_complex_matrix(cuDoubleComplex **matrix, long Nx, long Ny) {
   cuDoubleComplex *d_matrix;

   if(cudaHostGetDevicePointer((void **) &d_matrix, matrix[0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for complex matrix in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_matrix, Ny * sizeof(cuDoubleComplex), Ny * sizeof(cuDoubleComplex), Nx);
}

/**
 *    Map pinned double tensor
 */
struct cudaPitchedPtr map_double_tensor(double ***tensor, long Nx, long Ny, long Nz) {
   double *d_tensor;

   if(cudaHostGetDevicePointer((void **) &d_tensor, tensor[0][0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for double tensor in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_tensor, Nz * sizeof(double), Nz * sizeof(double), Ny);
}

/**
 *    Map pinned complex tensor
 */
struct cudaPitchedPtr map_complex_tensor(cuDoubleComplex ***tensor, long Nx, long Ny, long Nz) {
   cuDoubleComplex *d_tensor;

   if(cudaHostGetDevicePointer((void **) &d_tensor, tensor[0][0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for complex tensor in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_tensor, Nz * sizeof(cuDoubleComplex), Nz * sizeof(cuDoubleComplex), Ny);
}

/* Pinned/mapped memory release */

/**
 *    Free pinned/mapped double matrix
 */
void free_pinned_double_matrix(double **matrix) {
   if (cudaHostUnregister(matrix[0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   free_double_matrix(matrix);
}

/**
 *    Free pinned/mapped complex matrix
 */
void free_pinned_complex_matrix(cuDoubleComplex **matrix) {
   if (cudaHostUnregister(matrix[0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   free_complex_matrix(matrix);
}

/**
 *    Free pinned/mapped double tensor
 */
void free_pinned_double_tensor(double ***tensor) {
   if (cudaHostUnregister(tensor[0][0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   free_double_tensor(tensor);
}

/**
 *    Free pinned/mapped complex tensor
 */
void free_pinned_complex_tensor(cuDoubleComplex ***tensor) {
   if (cudaHostUnregister(tensor[0][0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   free_complex_tensor(tensor);
}

/* CUDA memory functions */

/**
 *    Double vector allocation on CUDA device
 */
double *alloc_double_vector_device(long Nx) {
   double *vector;

   if(cudaMalloc((void**) &vector, Nx * sizeof(double)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA double vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return vector;
}

/**
 *    Complex vector allocation on CUDA device
 */
cuDoubleComplex *alloc_complex_vector_device(long Nx) {
   cuDoubleComplex *vector;

   if(cudaMalloc((void**) &vector, Nx * sizeof(cuDoubleComplex)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA complex vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return vector;
}

/**
 *    Double matrix allocation on CUDA device
 */
struct cudaPitchedPtr alloc_double_matrix_device(long Nx, long Ny) {
   struct cudaPitchedPtr matrix;

   if(cudaMalloc3D(&matrix, make_cudaExtent(Ny * sizeof(double), Nx, 1)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return matrix;
}

/**
 *    Complex matrix allocation on CUDA device
 */
struct cudaPitchedPtr alloc_complex_matrix_device(long Nx, long Ny) {
   struct cudaPitchedPtr matrix;

   if(cudaMalloc3D(&matrix, make_cudaExtent(Ny * sizeof(cuDoubleComplex), Nx, 1)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return matrix;
}

/**
 *    Double tensor allocation on CUDA device
 */
struct cudaPitchedPtr alloc_double_tensor_device(long Nx, long Ny, long Nz) {
   struct cudaPitchedPtr tensor;

   if(cudaMalloc3D(&tensor, make_cudaExtent(Nz * sizeof(double), Ny, Nx)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return tensor;
}

/**
 *    Complex tensor allocation on CUDA device
 */
struct cudaPitchedPtr alloc_complex_tensor_device(long Nx, long Ny, long Nz) {
   struct cudaPitchedPtr tensor;

   if(cudaMalloc3D(&tensor, make_cudaExtent(Nz * sizeof(cuDoubleComplex), Ny, Nx)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return tensor;
}

/**
 *    Free double vector on CUDA device
 */
void free_double_vector_device(double *vector) {
   if (cudaFree(vector) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for double vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free complex vector on CUDA device
 */
void free_complex_vector_device(cuDoubleComplex *vector) {
   if (cudaFree(vector) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for complex vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free double matrix on CUDA device
 */
void free_double_matrix_device(struct cudaPitchedPtr matrix) {
   if (cudaFree(matrix.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free complex matrix on CUDA device
 */
void free_complex_matrix_device(struct cudaPitchedPtr matrix) {
   if (cudaFree(matrix.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free double tensor on CUDA device
 */
void free_double_tensor_device(struct cudaPitchedPtr tensor) {
   if (cudaFree(tensor.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free complex tensor on CUDA device
 */
void free_complex_tensor_device(struct cudaPitchedPtr tensor) {
   if (cudaFree(tensor.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}
