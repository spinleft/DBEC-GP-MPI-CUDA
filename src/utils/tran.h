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

#ifndef TRAN_H
#define TRAN_H

#include <stdio.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include "mem.h"

#ifdef __cplusplus
extern "C" {
#endif

struct tran_params {
   void *orig_buf, *tran_buf;
   int *orig_cnts, *tran_cnts;
   int *orig_displ, *tran_displ;
   MPI_Datatype *orig_types, *tran_types;
   int nprocs;
   MPI_Request *send_req, *recv_req;
};

struct tran_params init_transpose_double(int, long, long, long, long, void *, size_t, void *, size_t);
struct tran_params init_transpose_complex(int, long, long, long, long, void *, size_t, void *, size_t);

void transpose(struct tran_params);
void transpose_back(struct tran_params);

void free_transpose(struct tran_params);

#ifdef __cplusplus
}
#endif

#endif /* TRAN_H */
