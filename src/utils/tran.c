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

#include "tran.h"

struct tran_params init_transpose_double(int nprocs, long localNx, long localNy, long Ny, long Nz, void *send_buf, size_t send_pitch, void *recv_buf,  size_t recv_pitch) {
   long cnti;
   MPI_Datatype tran_double_type;
   MPI_Datatype lxyz_double_type, xlyz_double_type;
   int *orig_cnts, *tran_cnts;
   int *orig_displ, *tran_displ;
   MPI_Datatype *orig_types, *tran_types;
   MPI_Request *send_req, *recv_req;
   struct tran_params tran;

   MPI_Type_contiguous(localNy * (send_pitch / sizeof(double)), MPI_DOUBLE, &tran_double_type);
   MPI_Type_commit(&tran_double_type);

   MPI_Type_create_hvector(localNx, 1, Ny * send_pitch, tran_double_type, &lxyz_double_type);
   MPI_Type_create_hvector(localNx, 1, localNy * recv_pitch, tran_double_type, &xlyz_double_type);
   MPI_Type_commit(&lxyz_double_type);
   MPI_Type_commit(&xlyz_double_type);

   orig_cnts = (int *) malloc(nprocs * sizeof(int));
   tran_cnts = (int *) malloc(nprocs * sizeof(int));

   orig_displ = (int *) malloc(nprocs * sizeof(int));
   tran_displ = (int *) malloc(nprocs * sizeof(int));

   orig_types = (MPI_Datatype *) malloc(nprocs * sizeof(MPI_Datatype));
   tran_types = (MPI_Datatype *) malloc(nprocs * sizeof(MPI_Datatype));

   send_req = (MPI_Request *) malloc(nprocs * sizeof(MPI_Request));
   recv_req = (MPI_Request *) malloc(nprocs * sizeof(MPI_Request));

   for (cnti = 0; cnti < nprocs; cnti ++) {
      orig_cnts[cnti] = 1;
      tran_cnts[cnti] = 1;

      orig_displ[cnti] = cnti * localNy * send_pitch;
      tran_displ[cnti] = cnti * localNx * localNy * recv_pitch;

      orig_types[cnti] = lxyz_double_type;
      tran_types[cnti] = xlyz_double_type;
   }

   tran.nprocs = nprocs;
   tran.orig_buf = send_buf;
   tran.tran_buf = recv_buf;
   tran.orig_cnts = orig_cnts;
   tran.tran_cnts = tran_cnts;
   tran.orig_displ = orig_displ;
   tran.tran_displ = tran_displ;
   tran.orig_types = orig_types;
   tran.tran_types = tran_types;
   tran.send_req = send_req;
   tran.recv_req = recv_req;

   return tran;
}

struct tran_params init_transpose_complex(int nprocs, long localNx, long localNy, long Ny, long Nz, void *send_buf, size_t send_pitch, void *recv_buf,  size_t recv_pitch) {
   long cnti;
   MPI_Datatype tran_complex_type;
   MPI_Datatype lxyz_complex_type, xlyz_complex_type;
   int *orig_send_cnts, *tran_recv_cnts;
   int *orig_displ, *tran_displ;
   MPI_Datatype *orig_types, *tran_types;
   MPI_Request *send_req, *recv_req;
   struct tran_params tran;

   MPI_Type_contiguous(localNy * (send_pitch / sizeof(cuDoubleComplex)), MPI_C_DOUBLE_COMPLEX, &tran_complex_type);
   MPI_Type_commit(&tran_complex_type);

   MPI_Type_create_hvector(localNx, 1, Ny * send_pitch, tran_complex_type, &lxyz_complex_type);
   MPI_Type_create_hvector(localNx, 1, localNy * recv_pitch, tran_complex_type, &xlyz_complex_type);
   MPI_Type_commit(&lxyz_complex_type);
   MPI_Type_commit(&xlyz_complex_type);

   orig_send_cnts = (int *) malloc(nprocs * sizeof(int));
   tran_recv_cnts = (int *) malloc(nprocs * sizeof(int));

   orig_displ = (int *) malloc(nprocs * sizeof(int));
   tran_displ = (int *) malloc(nprocs * sizeof(int));

   orig_types = (MPI_Datatype *) malloc(nprocs * sizeof(MPI_Datatype));
   tran_types = (MPI_Datatype *) malloc(nprocs * sizeof(MPI_Datatype));

   send_req = (MPI_Request *) malloc(nprocs * sizeof(MPI_Request));
   recv_req = (MPI_Request *) malloc(nprocs * sizeof(MPI_Request));

   for (cnti = 0; cnti < nprocs; cnti ++) {
      orig_send_cnts[cnti] = 1;
      tran_recv_cnts[cnti] = 1;

      orig_displ[cnti] = cnti * localNy * send_pitch;
      tran_displ[cnti] = cnti * localNx * localNy * recv_pitch;

      orig_types[cnti] = lxyz_complex_type;
      tran_types[cnti] = xlyz_complex_type;
   }

   tran.nprocs = nprocs;
   tran.orig_buf = send_buf;
   tran.tran_buf = recv_buf;
   tran.orig_cnts = orig_send_cnts;
   tran.tran_cnts = tran_recv_cnts;
   tran.orig_displ = orig_displ;
   tran.tran_displ = tran_displ;
   tran.orig_types = orig_types;
   tran.tran_types = tran_types;
   tran.send_req = send_req;
   tran.recv_req = recv_req;

   return tran;
}

void transpose(struct tran_params tran) {
   long cnti;

   for (cnti = 0; cnti < tran.nprocs; cnti++) {
      MPI_Irecv(((char *)tran.tran_buf) + tran.tran_displ[cnti], 1, tran.tran_types[cnti], cnti, 0, MPI_COMM_WORLD, &(tran.recv_req[cnti]));
      MPI_Isend(((char *)tran.orig_buf) + tran.orig_displ[cnti], 1, tran.orig_types[cnti], cnti, 0, MPI_COMM_WORLD, &(tran.send_req[cnti]));
   }
   for (cnti = 0; cnti < tran.nprocs; cnti++) {
      MPI_Wait(&(tran.recv_req[cnti]), MPI_STATUSES_IGNORE);
      MPI_Wait(&(tran.send_req[cnti]), MPI_STATUSES_IGNORE);
   }

   //MPI_Alltoallw(tran.orig_buf, tran.orig_cnts, tran.orig_displ, tran.orig_types, tran.tran_buf, tran.tran_cnts, tran.tran_displ, tran.tran_types, MPI_COMM_WORLD);
}

void transpose_back(struct tran_params tran) {
   long cnti;

   for (cnti = 0; cnti < tran.nprocs; cnti++) {
      MPI_Irecv(((char *)tran.orig_buf) + tran.orig_displ[cnti], 1, tran.orig_types[cnti], cnti, 0, MPI_COMM_WORLD, &(tran.recv_req[cnti]));
      MPI_Isend(((char *)tran.tran_buf) + tran.tran_displ[cnti], 1, tran.tran_types[cnti], cnti, 0, MPI_COMM_WORLD, &(tran.send_req[cnti]));
   }
   for (cnti = 0; cnti < tran.nprocs; cnti++) {
      MPI_Wait(&(tran.recv_req[cnti]), MPI_STATUSES_IGNORE);
      MPI_Wait(&(tran.send_req[cnti]), MPI_STATUSES_IGNORE);
   }

   //MPI_Alltoallw(tran.tran_buf, tran.tran_cnts, tran.tran_displ, tran.tran_types, tran.orig_buf, tran.orig_cnts, tran.orig_displ, tran.orig_types, MPI_COMM_WORLD);
}

void free_transpose(struct tran_params tran) {
   free(tran.orig_cnts);
   free(tran.tran_cnts);

   free(tran.orig_displ);
   free(tran.tran_displ);

   free(tran.orig_types);
   free(tran.tran_types);

   free(tran.send_req);
   free(tran.recv_req);
}
