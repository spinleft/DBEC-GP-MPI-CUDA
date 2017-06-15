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

extern "C" {
#include "diffint.cuh"
}

/**
 *    Spatial 1D integration with Simpson's rule.
 *    h - space step
 *    f - array with the function values
 *    N - number of integration points
 */
__device__ __host__ double simpint(double h, double *f, long N) {
   long cnti;
   double sum, sumi, sumj, sumk;

   sumi = 0.; sumj = 0.; sumk = 0.;

   for(cnti = 1; cnti < N - 1; cnti += 2) {
      sumi += f[cnti];
      sumj += f[cnti - 1];
      sumk += f[cnti + 1];
   }

   sum = sumj + 4. * sumi + sumk;
   if(N % 2 == 0) sum += (5. * f[N - 1] + 8. * f[N - 2] - f[N - 3]) / 4.;

   return sum * h / 3.;

   /*int c;
   long cnti;
   double sum;

   sum = f[0];
   for (cnti = 1; cnti < N - 1; cnti ++) {
      c = 2 + 2 * (cnti % 2);
      sum += c * f[cnti];
   }
   sum += f[N - 1];

   return sum * h / 3.;*/
}

double *sum, *sumi, *sumj, *sumk;
double *tsumi, *tsumj, *tsumk;

void simpint_init(long N) {
   sum = alloc_double_vector_device(1);
   sumi = alloc_double_vector_device(1);
   sumj = alloc_double_vector_device(1);
   sumk = alloc_double_vector_device(1);

   long tempsize = ceil(1. * N / CUDA_BLOCK_SIZE / 2);

   tsumi = alloc_double_vector_device(tempsize);
   tsumj = alloc_double_vector_device(tempsize);
   tsumk = alloc_double_vector_device(tempsize);
}

void simpint_destroy() {
   free_double_vector_device(sum);
   free_double_vector_device(sumi);
   free_double_vector_device(sumj);
   free_double_vector_device(sumk);

   free_double_vector_device(tsumi);
   free_double_vector_device(tsumj);
   free_double_vector_device(tsumk);
}

__global__ void simpint2d_kernel(double h, cudaPitchedPtr matrix, double *vector, long N1, long N2) {
   long cnti;
   double *matrixrow;

   for (cnti = TID_X; cnti < N1; cnti += GRID_STRIDE_X) {
      matrixrow = get_double_matrix_row(matrix, cnti);

      vector[cnti] = simpint(h, matrixrow, N2);
   }
}

__global__ void simpint3d_kernel(double h, cudaPitchedPtr tensor, cudaPitchedPtr matrix, long N1, long N2, long N3) {
   long cnti, cntj;
   double *tensorrow, *matrixrow;

   for (cnti = TID_Y; cnti < N1; cnti += GRID_STRIDE_Y) {
      matrixrow = get_double_matrix_row(matrix, cnti);

      for (cntj = TID_X; cntj < N2; cntj += GRID_STRIDE_X) {
         tensorrow = get_double_tensor_row(tensor, cnti, cntj);

         matrixrow[cntj] = simpint(h, tensorrow, N3);
      }
   }
}

double simpint_gpu(double h, double *f, long N) {
   double f_sum;

   dim3 int_dimBlock;
   dim3 int_dimGrid;
   int int_shmemSize;

   int_dimBlock.x = CUDA_BLOCK_SIZE;
   int_dimGrid.x = ceil(1. * N / int_dimBlock.x / 2);
   int_shmemSize = 3 * int_dimBlock.x * sizeof(double);

   simpint1d_kernel1<<<int_dimGrid, int_dimBlock, int_shmemSize>>>(tsumi, tsumj, tsumk, f + 1, f, f + 2, N - 2, 2);
   simpint1d_kernel1<<<1, int_dimBlock, int_shmemSize>>>(sumi, sumj, sumk, tsumi, tsumj, tsumk, int_dimGrid.x, 1);
   simpint1d_kernel2<<<1,1>>>(h, sum, sumi, sumj, sumk, f, N);

   cudaMemcpy(&f_sum, sum, 1 * sizeof(double), cudaMemcpyDeviceToHost);

   return f_sum;
}

__global__ void simpint1d_kernel1(double *sumi, double *sumj, double *sumk, double *ini, double *inj, double *ink, long N, int step) {
   extern __shared__ double psumi[];
   double *psumj = &psumi[blockDim.x];
   double *psumk = &psumi[2 * blockDim.x];
   double tsumi = 0.;
   double tsumj = 0.;
   double tsumk = 0.;

   int tid = threadIdx.x;
   long index = (blockIdx.x*blockDim.x + tid) * step;

   if (index < N) {
      tsumi += ini[index];
      tsumj += inj[index];
      tsumk += ink[index];
   }

   psumi[tid] = tsumi;
   psumj[tid] = tsumj;
   psumk[tid] = tsumk;

   __syncthreads();

   // Start the shared memory loop on the next power of 2 less than the block size.
   // If block size is not a power of 2, accumulate the intermediate sums in the remainder range.
   int floorPow2 = blockDim.x;

   if (floorPow2 & (floorPow2-1)) {
      while (floorPow2 & (floorPow2-1)) {
         floorPow2 &= floorPow2-1;
      }
      if (tid >= floorPow2) {
         psumi[tid - floorPow2] += psumi[tid];
         psumj[tid - floorPow2] += psumj[tid];
         psumk[tid - floorPow2] += psumk[tid];
      }
      __syncthreads();
   }

   for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
      if (tid < activeThreads) {
         psumi[tid] += psumi[tid+activeThreads];
         psumj[tid] += psumj[tid+activeThreads];
         psumk[tid] += psumk[tid+activeThreads];
      }
      __syncthreads();
   }

   if (tid == 0) {
      sumi[blockIdx.x] = psumi[0];
      sumj[blockIdx.x] = psumj[0];
      sumk[blockIdx.x] = psumk[0];
   }
}

__global__ void simpint1d_kernel2(double h, double *sum, double *sumi, double *sumj, double *sumk, double *f, long N) {
   *sum = *sumj + 4. * *sumi + *sumk;
   if(N % 2 == 0) *sum += (5. * f[N - 1] + 8. * f[N - 2] - f[N - 3]) / 4.;
   *sum = *sum * h / 3.;
}

/**
 *    Richardson extrapolation formula for calculation of space derivatives.
 *    h  - space step
 *    f  - array with the function values
 *    df - array with the first derivatives of the function
 *    N  - number of space mesh points
 */
void diff(double h, double *f, double *df, long N) {
   dim3 dimBlock(CUDA_BLOCK_SIZE);
   dim3 dimGrid(ceil(1. * N/dimBlock.x));

   diff_kernel<<<dimBlock,dimGrid>>>(h, f, df, N);
}

__global__ void diff_kernel(double h, double *f, double *df, long N) {
   long index = threadIdx.x + blockDim.x * blockIdx.x;

   if (index == 0) {
      df[index] = 0.;
   }

   if (index == 1) {
      df[index] = (f[2] - f[0]) / (2. * h);
   }

   if (index > 1 && index < N - 2) {
      df[index] = (f[index - 2] - 8. * f[index - 1] + 8. * f[index + 1] - f[index + 2]) / (12. * h);
   }

   if (index == N - 2) {
      df[index] = (f[N - 1] - f[N - 3]) / (2. * h);
   }

   if (index == N - 1) {
      df[index] = 0.;
   }

}

/**
 *    Gauss-Legendre N-point quadrature formula.
 */
void gauleg(double x1, double x2, double *x, double *w, long N) {
   long m, j, i;
   double z1, z, xm, xl, pp, p3, p2, p1;

   m = (N + 1) / 2;
   xm = 0.5 * (x2 + x1);
   xl = 0.5 * (x2 - x1);
   for(i = 1; i <= m; i ++) {
      z = cos(4. * atan(1.) * (i - 0.25) / (N + 0.5));
      do {
         p1 = 1.;
         p2 = 0.;
         for(j = 1; j <= N; j ++) {
            p3 = p2;
            p2 = p1;
            p1 = ((2. * j - 1.) * z * p2 - (j - 1.) * p3) / j;
         }
         pp = N * (z * p1 - p2) / (z * z - 1.);
         z1 = z;
         z = z1 - p1 / pp;
      } while (fabs(z - z1) > GAULEG_EPS);
      x[i] = xm - xl * z;
      x[N + 1 - i] = xm + xl * z;
      w[i] = 2. * xl / ((1. - z * z) * pp * pp);
      w[N + 1 - i] = w[i];
   }

   return;
}
