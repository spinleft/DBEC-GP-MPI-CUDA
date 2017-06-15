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

#ifndef DIFFINT_H
#define DIFFINT_H

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cudautils.cuh"
#include "mem.h"

#define GAULEG_EPS   1.e-12

#ifdef __cplusplus
extern "C" {
#endif

void simpint_init(long);
void simpint_destroy();

__device__ __host__ double simpint(double, double *, long);
double simpint_gpu(double, double *, long);

void diff(double, double *, double *, long);
void gauleg(double, double, double *, double *, long);

__global__ void simpint2d_kernel(double, cudaPitchedPtr, double *, long, long);
__global__ void simpint3d_kernel(double, cudaPitchedPtr, cudaPitchedPtr, long, long, long);

__global__ void simpint1d_kernel1(double *, double *, double *, double *, double *, double *, long, int);
__global__ void simpint1d_kernel2(double, double *, double *, double *, double *, double *, long);
__global__ void diff_kernel(double, double *, double *, long);

#ifdef __cplusplus
}
#endif

#endif /* DIFFINT_H */
