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

#ifndef CUDATIMER_H
#define CUDATIMER_H

#include <cuda_runtime.h>
#include "../utils/cudautils.cuh"

void start_cuda_timer(cudaEvent_t start) {
   cudaCheckError(cudaEventRecord(start));
}

float stop_cuda_timer(cudaEvent_t start, cudaEvent_t stop) {
   float time;

   cudaCheckError(cudaEventRecord(stop));
   cudaCheckError(cudaEventSynchronize(stop));
   cudaCheckError(cudaEventElapsedTime(&time, start, stop));

   return time;
}

#endif /* CUDATIMER_H */
