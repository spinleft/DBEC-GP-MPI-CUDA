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

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>


#ifdef __cplusplus
extern "C" {
#endif

/* Host memory allocation */
double *alloc_double_vector(long);
cuDoubleComplex *alloc_complex_vector(long);
double **alloc_double_matrix(long, long);
cuDoubleComplex **alloc_complex_matrix(long, long);
double ***alloc_double_tensor(long, long, long);
cuDoubleComplex ***alloc_complex_tensor(long, long, long);

/* Host memory release */
void free_double_vector(double *);
void free_complex_vector(cuDoubleComplex *);
void free_double_matrix(double **);
void free_complex_matrix(cuDoubleComplex **);
void free_double_tensor(double ***);
void free_complex_tensor(cuDoubleComplex ***);

/* Pinned memory functions */
void pin_double_matrix(double **, long, long);
void pin_complex_matrix(cuDoubleComplex **, long, long);
void pin_double_tensor(double ***, long, long, long);
void pin_complex_tensor(cuDoubleComplex ***, long, long, long);

/* Mapped memory functions */
struct cudaPitchedPtr map_double_matrix(double **, long, long);
struct cudaPitchedPtr map_complex_matrix(cuDoubleComplex **, long, long);
struct cudaPitchedPtr map_double_tensor(double ***, long, long, long);
struct cudaPitchedPtr map_complex_tensor(cuDoubleComplex ***, long, long, long);

/* Pinned/mapped memory release */
void free_pinned_double_matrix(double **);
void free_pinned_complex_matrix(cuDoubleComplex **);
void free_pinned_double_tensor(double ***);
void free_pinned_complex_tensor(cuDoubleComplex ***);

/* CUDA memory allocation */
double *alloc_double_vector_device(long);
cuDoubleComplex *alloc_complex_vector_device(long);
struct cudaPitchedPtr alloc_double_matrix_device(long, long);
struct cudaPitchedPtr alloc_complex_matrix_device(long, long);
struct cudaPitchedPtr alloc_double_tensor_device(long, long, long);
struct cudaPitchedPtr alloc_complex_tensor_device(long, long, long);

/* CUDA memory release */
void free_double_vector_device(double *);
void free_complex_vector_device(cuDoubleComplex *);
void free_double_matrix_device(struct cudaPitchedPtr);
void free_complex_matrix_device(struct cudaPitchedPtr);
void free_double_tensor_device(struct cudaPitchedPtr);
void free_complex_tensor_device(struct cudaPitchedPtr);

#ifdef __cplusplus
}
#endif
