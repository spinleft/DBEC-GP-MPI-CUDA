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
#include <assert.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <mpi.h>
#include "../utils/mem.h"
#include "../utils/cfg.h"
#include "../utils/tran.h"
#include "../utils/diffint.cuh"
#include "../utils/cudautils.cuh"
#include "../utils/cudatimer.cuh"
#include <complex.h>

#define MAX(a, b, c)       (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c)
#define MAX_FILENAME_SIZE  256
#define RMS_ARRAY_SIZE     4

#define BOHR_RADIUS        5.2917720859e-11

char *output, *rmsout, *initout, *Nstpout, *Npasout, *Nrunout, *dynaout;
long outstpx, outstpy, outstpz, outstpt;

int rank, nprocs;
int opt;
long Na;
long Nstp, Npas, Nrun;
long Nx, Ny, Nz;
long Nx2, Ny2, Nz2;
long localNx, localNy, offsetNx, offsetNy;
double g, g0, gpar, gd, gd0, gdpar;
double aho, as, add;
double dx, dy, dz;
double dx2, dy2, dz2;
double dt;
double vnu, vlambda, vgamma;
double par;
double pi, cutoff;
int device;
int potmem;

double *x, *y, *z;
double *x2, *y2, *z2;
double ***pot;
double ***potdd;

cuDoubleComplex Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r, Ax, Ay, Az, minusAx, minusAy, minusAz;
cuDoubleComplex *calphax, *calphay, *calphaz;
cuDoubleComplex *cgammax, *cgammay, *cgammaz;

__constant__ long d_Nx, d_Ny, d_Nz;
__constant__ long d_Nx2, d_Ny2, d_Nz2;
__constant__ long d_localNx, d_localNy;
__constant__ double d_dx, d_dy, d_dz;
cudaPitchedPtr d_pot, d_potdd;
cudaMemcpy3DParms potparam = { 0 }, potddparam = { 0 };
cudaPitchedPtr d_dtensor1, d_dtensor2;
double *d_x2, *d_y2, *d_z2;
__constant__ double d_dt;
__constant__ cuDoubleComplex d_Ax0r, d_Ax, d_minusAx, d_Ay0r, d_Ay, d_minusAy, d_Az0r, d_Az, d_minusAz;
cuDoubleComplex *d_calphax, *d_calphay, *d_calphaz;
cuDoubleComplex *d_cgammax, *d_cgammay, *d_cgammaz;

struct tran_params tran_psi, tran_dpsi, tran_dd2;

cufftHandle planFWR, planFWC, planBWR, planBWC;

dim3 dimGrid1d, dimBlock1d;
dim3 dimGrid2d, dimBlock2d;
dim3 dimGrid3d, dimBlock3d;

cudaStream_t exec_stream;
cudaStream_t data_stream;

void readpar(void);
void initsize(long *, long *, long *);
void initpsi(cuDoubleComplex ***, double *);
void initpot(void);
void gencoef(void);
void initpotdd(double *, double *, double *, double *, double *, double *);
void initdevice(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr);

void calcpsi2(cudaPitchedPtr, cudaPitchedPtr);
void calcnorm(double *, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double *);
void calcmuen(double *, double *, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double *);
void (*calcpsidd2)(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr);
void calcpsidd2_mem(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr);
void calcpsidd2_cpy(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr);
void calcrms(double *, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double *, double *);
void calcnu(cudaPitchedPtr, cudaPitchedPtr);
void calclux(cudaPitchedPtr, cudaPitchedPtr);
void calcluy(cudaPitchedPtr, cudaPitchedPtr);
void calcluz(cudaPitchedPtr, cudaPitchedPtr);

void outdenx(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double *, double *, MPI_File);
void outdeny(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double *, double *, MPI_File);
void outdenz(cudaPitchedPtr, cudaPitchedPtr, double *, double *, MPI_File);
void outdenxy(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double **, MPI_File);
void outdenxz(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double **, MPI_File);
void outdenyz(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double **, MPI_File);
void outpsi2xy(cudaPitchedPtr, cudaPitchedPtr, double **, MPI_File);
void outpsi2xz(cudaPitchedPtr, cudaPitchedPtr, double **, MPI_File);
void outpsi2yz(cudaPitchedPtr, cudaPitchedPtr, double **, MPI_File);
void outdenxyz(cudaPitchedPtr, cudaPitchedPtr, double ***, double *, MPI_File);

__global__ void calcpsi2_kernel(cudaPitchedPtr, cudaPitchedPtr);
__global__ void calcnorm_kernel(cudaPitchedPtr, double);
__global__ void calcmuen_kernel1x(cudaPitchedPtr, cudaPitchedPtr, double);
__global__ void calcmuen_kernel1y(cudaPitchedPtr, cudaPitchedPtr, double);
__global__ void calcmuen_kernel1z(cudaPitchedPtr, cudaPitchedPtr, double);
__global__ void calcmuen_kernel2(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double, double);
__global__ void calcpsidd2_kernel1(cudaPitchedPtr, cudaPitchedPtr);
__global__ void calcpsidd2_kernel2a(cudaPitchedPtr, cudaPitchedPtr);
__global__ void calcpsidd2_kernel2b(cudaPitchedPtr, long);
__global__ void calcrms_kernel1x(cudaPitchedPtr, cudaPitchedPtr, double *);
__global__ void calcrms_kernel1y(cudaPitchedPtr, cudaPitchedPtr, double *);
__global__ void calcrms_kernel1z(cudaPitchedPtr, cudaPitchedPtr, double *);
__global__ void calcnu_kernel(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double, double);
__global__ void calclux_kernel(cudaPitchedPtr, cudaPitchedPtr, cuDoubleComplex *, cuDoubleComplex *);
__global__ void calcluy_kernel(cudaPitchedPtr, cudaPitchedPtr, cuDoubleComplex *, cuDoubleComplex *);
__global__ void calcluz_kernel(cudaPitchedPtr, cudaPitchedPtr, cuDoubleComplex *, cuDoubleComplex *);

__global__ void outdenx_kernel(cudaPitchedPtr, cudaPitchedPtr, long);
__global__ void outdeny_kernel(cudaPitchedPtr, cudaPitchedPtr, long);
__global__ void outdenz_kernel(cudaPitchedPtr, cudaPitchedPtr, long);
__global__ void outdenxy_kernel(cudaPitchedPtr, cudaPitchedPtr, long, long);
__global__ void outdenxz_kernel(cudaPitchedPtr, cudaPitchedPtr, long, long);
__global__ void outdenyz_kernel(cudaPitchedPtr, cudaPitchedPtr, long, long);
__global__ void outpsi2xy_kernel(cudaPitchedPtr, cudaPitchedPtr, long, long);
__global__ void outpsi2xz_kernel(cudaPitchedPtr, cudaPitchedPtr, long, long);
__global__ void outpsi2yz_kernel(cudaPitchedPtr, cudaPitchedPtr, long, long);
__global__ void outdenxyz_kernel(cudaPitchedPtr, cudaPitchedPtr, long, long, long);
