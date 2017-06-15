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

#include "imag3d-mpicuda.cuh"

int main(int argc, char **argv) {
   FILE *out;
   FILE *filerms;
   MPI_File mpifile;
   char filename[MAX_FILENAME_SIZE];
   int rankNx2;
   long offsetNx2;
   long cnti;
   double norm, mu, en;
   double *rms;
   double ***psi, ***tmpxyz;
   double **tmpxy, **tmpxz, **tmpyz;
   double *tmpxi, *tmpyi, *tmpzi, *tmpxj, *tmpyj, *tmpzj;
   double psiNx2Ny2Nz2;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   cudaCheckError(cudaDeviceReset());

   cudaPitchedPtr d_psi, d_psi_t;
   cudaPitchedPtr d_dpsi, d_dpsi_t;
   cudaPitchedPtr d_psidd2, d_psidd2_t, d_psidd2tmp;
   double *d_tmpx, *d_tmpy;
   cudaPitchedPtr d_tmpxyz, d_tmpxzy, d_tmpyxz, d_tmpyzx;
   cudaPitchedPtr d_tmpmu, d_tmpen;
   cudaPitchedPtr d_tmpxy, d_tmpxz, d_tmpyx, d_tmpyz;
   cudaPitchedPtr d_cbeta, d_cbeta_t;
   cudaMemcpy3DParms psiparam = { 0 }, psiNx2Ny2Nz2param = { 0 };

   double wall_time, init_time, iter_time;
   cudaEvent_t start, stop, total;
   iter_time = 0.;

   cudaCheckError(cudaEventCreate(&start));
   cudaCheckError(cudaEventCreate(&stop));
   cudaCheckError(cudaEventCreate(&total));
   start_cuda_timer(total);

   pi = 3.14159265358979;

   if ((argc != 3) || (strcmp(*(argv + 1), "-i") != 0)) {
      fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
      MPI_Finalize(); exit(EXIT_FAILURE);
   }

   if (! cfg_init(argv[2])) {
      fprintf(stderr, "Wrong input parameter file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }

   readpar();

   if (device == -1) {
      cudaCheckError(cudaGetDevice(&device));
   }

   cudaCheckError(cudaSetDevice(device));

   assert(Nx % nprocs == 0);
   assert(Ny % nprocs == 0);

   localNx = Nx / nprocs;
   localNy = Ny / nprocs;
   offsetNx = rank * localNx;
   offsetNy = rank * localNy;

   Nx2 = Nx / 2; Ny2 = Ny / 2; Nz2 = Nz / 2;
   dx2 = dx * dx; dy2 = dy * dy; dz2 = dz * dz;

   rankNx2 = Nx2 / localNx;
   offsetNx2 = Nx2 % localNx;

   rms = alloc_double_vector(RMS_ARRAY_SIZE);

   x = alloc_double_vector(Nx);
   y = alloc_double_vector(Ny);
   z = alloc_double_vector(Nz);

   x2 = alloc_double_vector(Nx);
   y2 = alloc_double_vector(Ny);
   z2 = alloc_double_vector(Nz);

   pot = alloc_double_tensor(localNx, Ny, Nz);
   potdd = alloc_double_tensor(Nx, localNy, Nz);
   psi = alloc_double_tensor(localNx, Ny, Nz);
   tmpxyz = psi;

   calphax = alloc_double_vector(Nx - 1);
   calphay = alloc_double_vector(Ny - 1);
   calphaz = alloc_double_vector(Nz - 1);
   cgammax = alloc_double_vector(Nx - 1);
   cgammay = alloc_double_vector(Ny - 1);
   cgammaz = alloc_double_vector(Nz - 1);

   tmpxy = alloc_double_matrix(localNx, Ny);
   tmpxz = alloc_double_matrix(localNx, Nz);
   tmpyz = alloc_double_matrix((rank == rankNx2) ? Ny : localNy, Nz);

   tmpxi = alloc_double_vector(Nx);
   tmpyi = alloc_double_vector(Ny);
   tmpzi = alloc_double_vector(Nz);
   tmpxj = alloc_double_vector(Nx);
   tmpyj = alloc_double_vector(Ny);
   tmpzj = alloc_double_vector(Nz);

   pin_double_tensor(psi, localNx, Ny, Nz);
   pin_double_tensor(pot, localNx, Ny, Nz);
   pin_double_tensor(potdd, Nx, localNy, Nz);


   d_x2 = alloc_double_vector_device(Nx);
   d_y2 = alloc_double_vector_device(Ny);
   d_z2 = alloc_double_vector_device(Nz);

   d_psi = alloc_double_tensor_device(localNx, Ny, Nz);
   d_psi_t = alloc_double_tensor_device(Nx, localNy, Nz);
   d_dpsi = alloc_double_tensor_device(localNx, Ny, Nz);
   d_dpsi_t = alloc_double_tensor_device(Nx, localNy, Nz);
   d_psidd2 = alloc_complex_tensor_device(localNx, Ny, Nz2 + 1);
   d_psidd2_t = alloc_complex_tensor_device(Nx, localNy, Nz2 + 1);

   d_calphax = alloc_double_vector_device(Nx - 1);
   d_calphay = alloc_double_vector_device(Ny - 1);
   d_calphaz = alloc_double_vector_device(Nz - 1);
   d_cbeta = d_dpsi;
   d_cbeta_t = d_dpsi_t;
   d_cgammax = alloc_double_vector_device(Nx - 1);
   d_cgammay = alloc_double_vector_device(Ny - 1);
   d_cgammaz = alloc_double_vector_device(Nz - 1);

   d_tmpx = alloc_double_vector_device(Nx);
   d_tmpy = alloc_double_vector_device(Ny);

   d_tmpxyz = d_dpsi;
   d_tmpyxz = make_cudaPitchedPtr(d_tmpxyz.ptr, d_tmpxyz.pitch, Nz, Nx);
   d_tmpyzx = alloc_double_tensor_device(localNy, Nz, Nx);
   d_tmpxzy = alloc_double_tensor_device(localNx, Nz, Ny);

   d_tmpmu = make_cudaPitchedPtr(d_psi_t.ptr, d_psi_t.pitch, Nz, Ny);
   d_tmpen = make_cudaPitchedPtr(d_dpsi_t.ptr, d_dpsi_t.pitch, Nz, Ny);

   d_tmpxy = alloc_double_matrix_device(Nx, Ny);
   d_tmpxz = alloc_double_matrix_device(Nx, Nz);
   d_tmpyx = alloc_double_matrix_device(Ny, Nx);
   d_tmpyz = alloc_double_matrix_device(Ny, Nz);
   d_psidd2tmp = alloc_double_matrix_device(Ny, 2 * (Nz2 + 1));

   // Required for transpose to work correctly
   assert(d_psi.pitch == d_psi_t.pitch);
   assert(d_dpsi.pitch == d_dpsi_t.pitch);
   assert(d_psidd2.pitch == d_psidd2_t.pitch);
   assert(d_psidd2.pitch == d_psidd2tmp.pitch);

   simpint_init(MAX(Nx, Ny, Nz));


   if (rank == 0) {
      if (output != NULL) {
         sprintf(filename, "%s.txt", output);
         out = fopen(filename, "w");
      } else out = stdout;
   } else out = fopen("/dev/null", "w");

   if (rank == 0) {
      if (rmsout != NULL) {
         sprintf(filename, "%s.txt", rmsout);
         filerms = fopen(filename, "w");
      } else filerms = NULL;
   } else filerms = fopen("/dev/null", "w");

   if (opt == 2) par = 2.;
   else par = 1.;

   fprintf(out, " Imaginary time propagation 3D,   OPTION = %d, MPI_NUM_PROCS = %d\n\n", opt, nprocs);
   fprintf(out, "  Number of Atoms N = %li, Unit of length AHO = %.8f m\n", Na, aho);
   fprintf(out, "  Scattering length a = %.2f*a0, Dipolar ADD = %.2f*a0\n", as, add);
   fprintf(out, "  Nonlinearity G_3D = %.4f, Strength of DDI GD_3D = %.5f\n", g0, gd0);
   fprintf(out, "  Parameters of trap: GAMMA = %.2f, NU = %.2f, LAMBDA = %.2f\n\n", vgamma, vnu, vlambda);
   fprintf(out, " # Space Stp: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
   fprintf(out, "  Space Step: DX = %.6f, DY = %.6f, DZ = %.6f\n", dx, dy, dz);
   fprintf(out, " # Time Stp : NSTP = %li, NPAS = %li, NRUN = %li\n", Nstp, Npas, Nrun);
   fprintf(out, "   Time Step:   DT = %.6f\n",  dt);
   fprintf(out, "   Dipolar Cut off:   R = %.3f\n\n",  cutoff);
   fprintf(out, "                  --------------------------------------------------------\n");
   fprintf(out, "                    Norm      Chem        Ener/N     <r>     |Psi(0,0,0)|^2\n");
   fprintf(out, "                  --------------------------------------------------------\n");
   fflush(out);

   initpsi(psi);
   initpot();
   gencoef();
   initpotdd(tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
   initdevice(d_psi, d_psi_t, d_dpsi, d_dpsi_t, d_psidd2, d_psidd2_t);


   psiparam.srcPtr = make_cudaPitchedPtr(psi[0][0], Nz * sizeof(double), Nz, Ny);
   psiparam.dstPtr = d_psi;
   psiparam.extent = make_cudaExtent(Nz * sizeof(double), Ny, localNx);
   psiparam.kind = cudaMemcpyHostToDevice;

   psiNx2Ny2Nz2param.srcPtr = make_cudaPitchedPtr((double *)((((char *)d_psi.ptr) + offsetNx2 * d_psi.pitch * Ny) + Ny2 * d_psi.pitch) + Nz2, sizeof(double), 1, 1);
   psiNx2Ny2Nz2param.dstPtr = make_cudaPitchedPtr(&psiNx2Ny2Nz2, sizeof(double), 1, 1);
   psiNx2Ny2Nz2param.extent = make_cudaExtent(sizeof(double), 1, 1);
   psiNx2Ny2Nz2param.kind = cudaMemcpyDeviceToHost;

   cudaCheckError(cudaMemcpy3D(&psiparam));


   calcnorm(&norm, d_psi, d_tmpxyz, d_tmpxy, d_tmpx);
   calcmuen(&mu, &en, d_psi, d_psi_t, d_dpsi, d_dpsi_t, d_psidd2, d_psidd2_t, d_psidd2tmp, d_tmpmu, d_tmpen, d_tmpxy, d_tmpx);
   calcrms(rms, d_psi, d_psi_t, d_tmpyzx, d_tmpxzy, d_tmpxyz, d_tmpyz, d_tmpxz, d_tmpxy, d_tmpx, d_tmpy);


   cudaCheckError(cudaMemcpy3D(&psiNx2Ny2Nz2param));
   MPI_Bcast(&psiNx2Ny2Nz2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
   fprintf(out, "Initial : %15.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psiNx2Ny2Nz2 * psiNx2Ny2Nz2);
   fflush(out);

   if (initout != NULL) {
      sprintf(filename, "%s.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxyz(d_psi, d_tmpxyz, tmpxyz, tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_x.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenx(d_psi, d_tmpxyz, d_tmpxy, d_tmpx, tmpxi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_y.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdeny(d_psi_t, d_tmpyxz, d_tmpyx, d_tmpy, tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_z.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenz(d_psi, d_tmpxy, d_tmpx, tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xy.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxy(d_psi, d_tmpxyz, d_tmpxy, tmpxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xz.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxz(d_psi, d_tmpxzy, d_tmpxz, tmpxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_yz.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenyz(d_psi_t, d_tmpyzx, d_tmpyz, tmpyz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_xy0.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xy(d_psi, d_tmpxy, tmpxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_x0z.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xz(d_psi, d_tmpxz, tmpxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_0yz.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2yz(d_psi, d_tmpyz, tmpyz, mpifile);
      MPI_File_close(&mpifile);
   }
   if (rmsout != NULL) {
      fprintf(filerms, " Imaginary time propagation 3D,   OPTION = %d\n\n", opt);
      fprintf(filerms, "                  --------------------------------------------------------\n");
      fprintf(filerms, "Values of rms size:     <r>          <x>          <y>          <z>\n");
      fprintf(filerms, "                  --------------------------------------------------------\n");
      fprintf(filerms, "           Initial:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
      fflush(filerms);
   }

   if (Nstp != 0) {
      double g_stp = par * g0 / (double) Nstp;
      double gd_stp = par * gd0 / (double) Nstp;
      g = 0.;
      gd = 0.;
      start_cuda_timer(start);
      for (cnti = 0; cnti < Nstp; cnti ++) {
         g += g_stp;
         gd += gd_stp;

         calcpsidd2(d_psi, d_psi_t, d_psidd2, d_psidd2_t, d_tmpyz);
         calcnu(d_psi, d_psidd2);
         calclux(d_psi_t, d_cbeta_t);
         calcluy(d_psi, d_cbeta);
         calcluz(d_psi, d_cbeta);
         calcnorm(&norm, d_psi, d_tmpxyz, d_tmpxy, d_tmpx);
      }
      iter_time += stop_cuda_timer(start, stop);

      calcmuen(&mu, &en, d_psi, d_psi_t, d_dpsi, d_dpsi_t, d_psidd2, d_psidd2_t, d_psidd2tmp, d_tmpmu, d_tmpen, d_tmpxy, d_tmpx);
      calcrms(rms, d_psi, d_psi_t, d_tmpyzx, d_tmpxzy, d_tmpxyz, d_tmpyz, d_tmpxz, d_tmpxy, d_tmpx, d_tmpy);

      cudaCheckError(cudaMemcpy3D(&psiNx2Ny2Nz2param));
      MPI_Bcast(&psiNx2Ny2Nz2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
      fprintf(out, "After NSTP iter.:%8.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psiNx2Ny2Nz2 * psiNx2Ny2Nz2);
      fflush(out);
      if (rmsout != NULL) {
         fprintf(filerms, "  After NSTP iter.:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
         fflush(filerms);
      }
      if (Nstpout != NULL) {
         sprintf(filename, "%s.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxyz(d_psi, d_tmpxyz, tmpxyz, tmpzi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s1d_x.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenx(d_psi, d_tmpxyz, d_tmpxy, d_tmpx, tmpxi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s1d_y.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdeny(d_psi_t, d_tmpyxz, d_tmpyx, d_tmpy, tmpyi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s1d_z.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenz(d_psi, d_tmpxy, d_tmpx, tmpzi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s2d_xy.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxy(d_psi, d_tmpxyz, d_tmpxy, tmpxy, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s2d_xz.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxz(d_psi, d_tmpxzy, d_tmpxz, tmpxz, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s2d_yz.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenyz(d_psi_t, d_tmpyzx, d_tmpyz, tmpyz, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s3d_xy0.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2xy(d_psi, d_tmpxy, tmpxy, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s3d_x0z.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2xz(d_psi, d_tmpxz, tmpxz, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s3d_0yz.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2yz(d_psi, d_tmpyz, tmpyz, mpifile);
         MPI_File_close(&mpifile);
      }
   } else {
      g = par * g0;
      gd = par * gd0;
   }


   start_cuda_timer(start);
   for (cnti = 0; cnti < Npas; cnti ++) {
      calcpsidd2(d_psi, d_psi_t, d_psidd2, d_psidd2_t, d_tmpyz);
      calcnu(d_psi, d_psidd2);
      calclux(d_psi_t, d_cbeta_t);
      calcluy(d_psi, d_cbeta);
      calcluz(d_psi, d_cbeta);
      calcnorm(&norm, d_psi, d_tmpxyz, d_tmpxy, d_tmpx);
   }
   iter_time += stop_cuda_timer(start, stop);

   calcmuen(&mu, &en, d_psi, d_psi_t, d_dpsi, d_dpsi_t, d_psidd2, d_psidd2_t, d_psidd2tmp, d_tmpmu, d_tmpen, d_tmpxy, d_tmpx);
   calcrms(rms, d_psi, d_psi_t, d_tmpyzx, d_tmpxzy, d_tmpxyz, d_tmpyz, d_tmpxz, d_tmpxy, d_tmpx, d_tmpy);

   cudaCheckError(cudaMemcpy3D(&psiNx2Ny2Nz2param));
   MPI_Bcast(&psiNx2Ny2Nz2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
   fprintf(out, "After NPAS iter.:%8.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psiNx2Ny2Nz2 *  psiNx2Ny2Nz2);
   fflush(out);

   if (Npasout != NULL) {
      sprintf(filename, "%s.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxyz(d_psi, d_tmpxyz, tmpxyz, tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_x.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenx(d_psi, d_tmpxyz, d_tmpxy, d_tmpx, tmpxi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_y.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdeny(d_psi_t, d_tmpyxz, d_tmpyx, d_tmpy, tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_z.txt", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenz(d_psi, d_tmpxy, d_tmpx, tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xy.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxy(d_psi, d_tmpxyz, d_tmpxy, tmpxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xz.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxz(d_psi, d_tmpxzy, d_tmpxz, tmpxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_yz.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenyz(d_psi_t, d_tmpyzx, d_tmpyz, tmpyz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_xy0.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xy(d_psi, d_tmpxy, tmpxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_x0z.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xz(d_psi, d_tmpxz, tmpxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_0yz.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2yz(d_psi, d_tmpyz, tmpyz, mpifile);
      MPI_File_close(&mpifile);
   }
   if (rmsout != NULL) {
      fprintf(filerms, "  After NPAS iter.:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
      fflush(filerms);
   }

   start_cuda_timer(start);
   for (cnti = 0; cnti < Nrun; cnti ++) {
      calcpsidd2(d_psi, d_psi_t, d_psidd2, d_psidd2_t, d_tmpyz);
      calcnu(d_psi, d_psidd2);
      calclux(d_psi_t, d_cbeta_t);
      calcluy(d_psi, d_cbeta);
      calcluz(d_psi, d_cbeta);
      calcnorm(&norm, d_psi, d_tmpxyz, d_tmpxy, d_tmpx);
   }
   iter_time += stop_cuda_timer(start, stop);

   calcmuen(&mu, &en, d_psi, d_psi_t, d_dpsi, d_dpsi_t, d_psidd2, d_psidd2_t, d_psidd2tmp, d_tmpmu, d_tmpen, d_tmpxy, d_tmpx);
   calcrms(rms, d_psi, d_psi_t, d_tmpyzx, d_tmpxzy, d_tmpxyz, d_tmpyz, d_tmpxz, d_tmpxy, d_tmpx, d_tmpy);

   cudaCheckError(cudaMemcpy3D(&psiNx2Ny2Nz2param));
   MPI_Bcast(&psiNx2Ny2Nz2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
   fprintf(out, "After NRUN iter.:%8.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psiNx2Ny2Nz2 *  psiNx2Ny2Nz2);
   fflush(out);

   if (Nrunout != NULL) {
      sprintf(filename, "%s.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxyz(d_psi, d_tmpxyz, tmpxyz, tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_x.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenx(d_psi, d_tmpxyz, d_tmpxy, d_tmpx, tmpxi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_y.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdeny(d_psi_t, d_tmpyxz, d_tmpyx, d_tmpy, tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_z.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenz(d_psi, d_tmpxy, d_tmpx, tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xy.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxy(d_psi, d_tmpxyz, d_tmpxy, tmpxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xz.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxz(d_psi, d_tmpxzy, d_tmpxz, tmpxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_yz.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenyz(d_psi_t, d_tmpyzx, d_tmpyz, tmpyz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_xy0.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xy(d_psi, d_tmpxy, tmpxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_x0z.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xz(d_psi, d_tmpxz, tmpxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_0yz.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2yz(d_psi, d_tmpyz, tmpyz, mpifile);
      MPI_File_close(&mpifile);
   }
   if (rmsout != NULL) {
      fprintf(filerms, "  After NRUN iter.:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
      fprintf(filerms, "                  --------------------------------------------------------\n");
   }
   if (rmsout != NULL) fclose(filerms);

   fprintf(out, "                  --------------------------------------------------------\n\n");

   free_double_vector(rms);

   free_double_vector(x);
   free_double_vector(y);
   free_double_vector(z);

   free_double_vector(x2);
   free_double_vector(y2);
   free_double_vector(z2);

   free_pinned_double_tensor(pot);
   free_pinned_double_tensor(potdd);
   free_pinned_double_tensor(psi);

   free_double_vector(calphax);
   free_double_vector(calphay);
   free_double_vector(calphaz);
   free_double_vector(cgammax);
   free_double_vector(cgammay);
   free_double_vector(cgammaz);

   free_double_vector(tmpxi);
   free_double_vector(tmpyi);
   free_double_vector(tmpzi);
   free_double_vector(tmpxj);
   free_double_vector(tmpyj);
   free_double_vector(tmpzj);

   free_double_matrix(tmpxy);
   free_double_matrix(tmpxz);
   free_double_matrix(tmpyz);

   cudaCheckError(cudaStreamDestroy(exec_stream));
   cudaCheckError(cudaStreamDestroy(data_stream));

   free_double_vector_device(d_x2);
   free_double_vector_device(d_y2);
   free_double_vector_device(d_z2);

   free_double_tensor_device(d_psi);
   free_double_tensor_device(d_psi_t);
   free_double_tensor_device(d_dpsi);
   free_double_tensor_device(d_dpsi_t);
   free_complex_tensor_device(d_psidd2);
   free_complex_tensor_device(d_psidd2_t);

   free_double_tensor_device(d_tmpyzx);
   free_double_tensor_device(d_tmpxzy);

   free_double_vector_device(d_calphax);
   free_double_vector_device(d_calphay);
   free_double_vector_device(d_calphaz);
   free_double_vector_device(d_cgammax);
   free_double_vector_device(d_cgammay);
   free_double_vector_device(d_cgammaz);

   if (d_dtensor1.ptr != NULL) free_double_tensor_device(d_dtensor1);
   if (d_dtensor2.ptr != NULL) free_double_tensor_device(d_dtensor2);

   free_double_vector_device(d_tmpx);
   free_double_vector_device(d_tmpy);

   free_double_matrix_device(d_tmpxy);
   free_double_matrix_device(d_tmpxz);
   free_double_matrix_device(d_tmpyx);
   free_double_matrix_device(d_tmpyz);
   free_double_matrix_device(d_psidd2tmp);

   simpint_destroy();

   free_transpose(tran_psi);
   free_transpose(tran_dpsi);
   free_transpose(tran_dd2);

   cudaCheckFFTError(cufftDestroy(planFWR));
   cudaCheckFFTError(cufftDestroy(planFWC));
   cudaCheckFFTError(cufftDestroy(planBWR));
   cudaCheckFFTError(cufftDestroy(planBWC));

   wall_time = stop_cuda_timer(total, stop);
   init_time = wall_time - iter_time;
   fprintf(out, " Initialization/allocation wall-clock time: %.3f seconds\n", init_time / 1000.);
   fprintf(out, " Calculation (iterations) wall-clock time: %.3f seconds\n", iter_time / 1000.);

   if(output != NULL) fclose(out);

   cudaCheckError(cudaEventDestroy(start));
   cudaCheckError(cudaEventDestroy(stop));
   cudaCheckError(cudaEventDestroy(total));

   cudaDeviceSynchronize();
   cudaCheckError(cudaDeviceReset());

   MPI_Finalize();

   return(EXIT_SUCCESS);
}

/**
 *    Reading input parameters from the configuration file.
 */
void readpar(void) {
   char *cfg_tmp;

   if ((cfg_tmp = cfg_read("OPTION")) == NULL) {
      fprintf(stderr, "OPTION is not defined in the configuration file\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   opt = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NATOMS")) == NULL) {
      fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   Na = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("G0")) == NULL) {
      if ((cfg_tmp = cfg_read("AHO")) == NULL) {
         fprintf(stderr, "AHO is not defined in the configuration file.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
      aho = atof(cfg_tmp);

      if ((cfg_tmp = cfg_read("AS")) == NULL) {
         fprintf(stderr, "AS is not defined in the configuration file.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
      as = atof(cfg_tmp);

      g0 = 4. * pi * as * Na * BOHR_RADIUS / aho;
   } else {
      g0 = atof(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("GDD0")) == NULL) {
      if ((cfg_tmp = cfg_read("AHO")) == NULL) {
         fprintf(stderr, "AHO is not defined in the configuration file.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
      aho = atof(cfg_tmp);

      if ((cfg_tmp = cfg_read("ADD")) == NULL) {
         fprintf(stderr, "ADD is not defined in the configuration file.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
      add = atof(cfg_tmp);

      gd0 = 3. * add * Na * BOHR_RADIUS / aho;
   } else {
      gd0 = atof(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("NX")) == NULL) {
      fprintf(stderr, "NX is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   Nx = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NY")) == NULL) {
      fprintf(stderr, "NY is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   Ny = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NZ")) == NULL) {
      fprintf(stderr, "Nz is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   Nz = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("DX")) == NULL) {
      fprintf(stderr, "DX is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   dx = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DY")) == NULL) {
      fprintf(stderr, "DY is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   dy = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DZ")) == NULL) {
      fprintf(stderr, "DZ is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   dz = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DT")) == NULL) {
      fprintf(stderr, "DT is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   dt = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("GAMMA")) == NULL) {
      fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   vgamma = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NU")) == NULL) {
      fprintf(stderr, "NU is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   vnu = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("LAMBDA")) == NULL) {
      fprintf(stderr, "LAMBDA is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   vlambda = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NSTP")) == NULL) {
      fprintf(stderr, "NSTP is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   Nstp = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NPAS")) == NULL) {
      fprintf(stderr, "NPAS is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   Npas = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NRUN")) == NULL) {
      fprintf(stderr, "NRUN is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   Nrun = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("CUTOFF")) == NULL) {
      fprintf(stderr, "CUTOFF is not defined in the configuration file.\n");
      MPI_Finalize(); exit(EXIT_FAILURE);
   }
   cutoff = atof(cfg_tmp);

   output = cfg_read("OUTPUT");
   rmsout = cfg_read("RMSOUT");
   initout = cfg_read("INITOUT");
   Nstpout = cfg_read("NSTPOUT");
   Npasout = cfg_read("NPASOUT");
   Nrunout = cfg_read("NRUNOUT");

   if ((initout != NULL) || (Nstpout != NULL) || (Npasout != NULL) || (Nrunout != NULL)) {
      if ((cfg_tmp = cfg_read("OUTSTPX")) == NULL) {
         fprintf(stderr, "OUTSTPX is not defined in the configuration file.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
      outstpx = atol(cfg_tmp);

      if ((cfg_tmp = cfg_read("OUTSTPY")) == NULL) {
         fprintf(stderr, "OUTSTPY is not defined in the configuration file.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
      outstpy = atol(cfg_tmp);

      if ((cfg_tmp = cfg_read("OUTSTPZ")) == NULL) {
         fprintf(stderr, "OUTSTPZ is not defined in the configuration file.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
      outstpz = atol(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("DEVICE")) == NULL) {
      device = -1;
   } else {
      device = atoi(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("POTMEM")) == NULL) {
      potmem = 2;
   } else {
      potmem = atoi(cfg_tmp);
      if (potmem < 0 || potmem > 2) {
         fprintf(stderr, "POTMEM can be 0, 1 or 2.\n");
         MPI_Finalize(); exit(EXIT_FAILURE);
      }
   }

   return;
}

/**
 *    Initialization of the space mesh and the initial wave function.
 *    psi - array with the wave function values
 */
void initpsi(double ***psi) {
   long cnti, cntj, cntk;
   double cpsi;
   double tmp;

   cpsi = sqrt(pi * sqrt(pi / (vgamma * vnu * vlambda)));

   for (cnti = 0; cnti < Nx; cnti ++) {
      x[cnti] = (cnti - Nx2) * dx;
      x2[cnti] = x[cnti] * x[cnti];
   }

   for (cntj = 0; cntj < Ny; cntj ++) {
      y[cntj] = (cntj - Ny2) * dy;
      y2[cntj] = y[cntj] * y[cntj];
   }

   for (cntk = 0; cntk < Nz; cntk ++) {
      z[cntk] = (cntk - Nz2) * dz;
      z2[cntk] = z[cntk] * z[cntk];
   }

   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmp = exp(- 0.5 * (vgamma * x2[offsetNx + cnti] + vnu * y2[cntj] + vlambda * z2[cntk]));
            psi[cnti][cntj][cntk] = tmp / cpsi;
         }
      }
   }

   return;
}

/**
 *    Initialization of the potential.
 */
void initpot() {
   long cnti, cntj, cntk;
   double vnu2, vlambda2, vgamma2;

   vnu2 = vnu * vnu;
   vlambda2 = vlambda * vlambda;
   vgamma2 = vgamma * vgamma;

   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            pot[cnti][cntj][cntk] = 0.5 * par * (vgamma2 * x2[offsetNx + cnti] + vnu2 * y2[cntj] + vlambda2 * z2[cntk]);
         }
      }
   }

   return;
}

/**
 *    Crank-Nicolson scheme coefficients generation.
 */
void gencoef(void) {
   long cnti;

   Ax0 = 1. + dt / dx2 / (3. - par);
   Ay0 = 1. + dt / dy2 / (3. - par);
   Az0 = 1. + dt / dz2 / (3. - par);

   Ax0r = 1. - dt / dx2 / (3. - par);
   Ay0r = 1. - dt / dy2 / (3. - par);
   Az0r = 1. - dt / dz2 / (3. - par);

   Ax = - 0.5 * dt / dx2 / (3. - par);
   Ay = - 0.5 * dt / dy2 / (3. - par);
   Az = - 0.5 * dt / dz2 / (3. - par);

   calphax[Nx - 2] = 0.;
   cgammax[Nx - 2] = - 1. / Ax0;
   for (cnti = Nx - 2; cnti > 0; cnti --) {
      calphax[cnti - 1] = Ax * cgammax[cnti];
      cgammax[cnti - 1] = - 1. / (Ax0 + Ax * calphax[cnti - 1]);
   }

   calphay[Ny - 2] = 0.;
   cgammay[Ny - 2] = - 1. / Ay0;
   for (cnti = Ny - 2; cnti > 0; cnti --) {
      calphay[cnti - 1] = Ay * cgammay[cnti];
      cgammay[cnti - 1] = - 1. / (Ay0 + Ay * calphay[cnti - 1]);
   }

   calphaz[Nz - 2] = 0.;
   cgammaz[Nz - 2] = - 1. / Az0;
   for (cnti = Nz - 2; cnti > 0; cnti --) {
      calphaz[cnti - 1] = Az * cgammaz[cnti];
      cgammaz[cnti - 1] = - 1. / (Az0 + Az * calphaz[cnti - 1]);
   }

   return;
}

/**
 *    Initialization of the dipolar potential.
 *    kx  - array with the space mesh values in the x-direction in the K-space
 *    ky  - array with the space mesh values in the y-direction in the K-space
 *    kz  - array with the space mesh values in the z-direction in the K-space
 *    kx2 - array with the squared space mesh values in the x-direction in the
 *          K-space
 *    ky2 - array with the squared space mesh values in the y-direction in the
 *          K-space
 *    kz2 - array with the squared space mesh values in the z-direction in the
 *          K-space
 */
void initpotdd(double *kx, double *ky, double *kz, double *kx2, double *ky2, double *kz2) {
   long cnti, cntj, cntk;
   double dkx, dky, dkz, xk, tmp;

   dkx = 2. * pi / (Nx * dx);
   dky = 2. * pi / (Ny * dy);
   dkz = 2. * pi / (Nz * dz);

   for (cnti = 0; cnti < Nx2; cnti ++) kx[cnti] = cnti * dkx;
   for (cnti = 0; cnti < Nx2; cnti ++) kx[cnti + Nx2] = (cnti - Nx2) * dkx;
   for (cntj = 0; cntj < Ny2; cntj ++) ky[cntj] = cntj * dky;
   for (cntj = 0; cntj < Ny2; cntj ++) ky[cntj + Ny2] = (cntj - Ny2) * dky;
   for (cntk = 0; cntk < Nz2; cntk ++) kz[cntk] = cntk * dkz;
   for (cntk = 0; cntk < Nz2; cntk ++) kz[cntk + Nz2] = (cntk - Nz2) * dkz;

   for (cnti = 0; cnti < Nx; cnti ++) kx2[cnti] = kx[cnti] * kx[cnti];
   for (cntj = 0; cntj < localNy; cntj ++) ky2[cntj] = ky[offsetNy + cntj] * ky[offsetNy + cntj];
   for (cntk = 0; cntk < Nz; cntk ++) kz2[cntk] = kz[cntk] * kz[cntk];


   for (cnti = 0; cnti < Nx; cnti ++) {
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            xk = sqrt(kz2[cntk] + kx2[cnti] + ky2[cntj]);
            tmp = 1. + 3. * cos(xk * cutoff) / (xk * xk * cutoff * cutoff) - 3. * sin(xk * cutoff) / (xk * xk * xk * cutoff * cutoff * cutoff);
            potdd[cnti][cntj][cntk] = (4. * pi * (3. * kz2[cntk] / (kx2[cnti] + ky2[cntj] + kz2[cntk]) - 1.) / 3.) * tmp;
         }
      }
   }

   if (rank == 0) {
      potdd[0][0][0] = 0.;
   }

   return;
}

void initdevice(cudaPitchedPtr psi, cudaPitchedPtr psi_t, cudaPitchedPtr dpsi, cudaPitchedPtr dpsi_t, cudaPitchedPtr psidd2, cudaPitchedPtr psidd2_t) {
   int numSMs;
   size_t workSizeFWR, workSizeFWC, workSizeBWC, workSizeBWR, workSizeMax;
   void *workArea;

   cudaCheckError(cudaStreamCreate(&exec_stream));
   cudaCheckError(cudaStreamCreate(&data_stream));

   cudaCheckError(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(Nx)));
   cudaCheckError(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(Ny)));
   cudaCheckError(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(Nz)));

   cudaCheckError(cudaMemcpyToSymbol(d_Nx2, &Nx2, sizeof(Nx2)));
   cudaCheckError(cudaMemcpyToSymbol(d_Ny2, &Ny2, sizeof(Ny2)));
   cudaCheckError(cudaMemcpyToSymbol(d_Nz2, &Nz2, sizeof(Nz2)));

   cudaCheckError(cudaMemcpyToSymbol(d_localNx, &localNx, sizeof(localNx)));
   cudaCheckError(cudaMemcpyToSymbol(d_localNy, &localNy, sizeof(localNy)));

   cudaCheckError(cudaMemcpyToSymbol(d_dx, &dx, sizeof(dx)));
   cudaCheckError(cudaMemcpyToSymbol(d_dy, &dy, sizeof(dy)));
   cudaCheckError(cudaMemcpyToSymbol(d_dz, &dz, sizeof(dz)));

   cudaCheckError(cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt)));

   cudaCheckError(cudaMemcpyToSymbol(d_Ax0r, &Ax0r, sizeof(Ax0r)));
   cudaCheckError(cudaMemcpyToSymbol(d_Ax, &Ax, sizeof(Ax)));
   cudaCheckError(cudaMemcpyToSymbol(d_Ay0r, &Ay0r, sizeof(Ay0r)));
   cudaCheckError(cudaMemcpyToSymbol(d_Ay, &Ay, sizeof(Ay)));
   cudaCheckError(cudaMemcpyToSymbol(d_Az0r, &Az0r, sizeof(Az0r)));
   cudaCheckError(cudaMemcpyToSymbol(d_Az, &Az, sizeof(Az)));

   cudaCheckError(cudaMemcpy(d_x2, x2, Nx * sizeof(double), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_y2, y2, Ny * sizeof(double), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_z2, z2, Nz * sizeof(double), cudaMemcpyHostToDevice));

   cudaCheckError(cudaMemcpy(d_calphax, calphax, (Nx - 1) * sizeof(double), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_calphay, calphay, (Ny - 1) * sizeof(double), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_calphaz, calphaz, (Nz - 1) * sizeof(double), cudaMemcpyHostToDevice));

   cudaCheckError(cudaMemcpy(d_cgammax, cgammax, (Nx - 1) * sizeof(double), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_cgammay, cgammay, (Ny - 1) * sizeof(double), cudaMemcpyHostToDevice));
   cudaCheckError(cudaMemcpy(d_cgammaz, cgammaz, (Nz - 1) * sizeof(double), cudaMemcpyHostToDevice));

   potparam.srcPtr = make_cudaPitchedPtr(pot[0][0], Nz * sizeof(double), Nz, Ny);
   //potparam.dstPtr = d_pot;
   potparam.extent = make_cudaExtent(Nz * sizeof(double), Ny, localNx);
   potparam.kind = cudaMemcpyHostToDevice;

   potddparam.srcPtr = make_cudaPitchedPtr(potdd[0][0], Nz * sizeof(double), Nz, localNy);
   //potddparam.dstPtr = d_potdd;
   potddparam.extent = make_cudaExtent(Nz * sizeof(double), localNy, Nx);
   potddparam.kind = cudaMemcpyHostToDevice;


   cudaCheckFFTError(cufftCreate(&planFWR));
   cudaCheckFFTError(cufftCreate(&planFWC));
   cudaCheckFFTError(cufftCreate(&planBWC));
   cudaCheckFFTError(cufftCreate(&planBWR));

   cudaCheckFFTError(cufftSetAutoAllocation(planFWR, 0));
   cudaCheckFFTError(cufftSetAutoAllocation(planFWC, 0));
   cudaCheckFFTError(cufftSetAutoAllocation(planBWC, 0));
   cudaCheckFFTError(cufftSetAutoAllocation(planBWR, 0));

   switch (potmem) {
   case 0:
      d_pot = map_double_tensor(pot, localNx, Ny, Nz);
      d_potdd = map_double_tensor(potdd, Nx, localNy, Nz);

      calcpsidd2 = &calcpsidd2_mem;

      break;
   case 1:
      d_dtensor1 = alloc_double_tensor_device(localNx, Ny, Nz);
      d_pot = d_dtensor1;
      d_potdd = make_cudaPitchedPtr((char *) d_dtensor1.ptr, d_dtensor1.pitch, Nz, localNy);

      potparam.dstPtr = d_pot;
      potddparam.dstPtr = d_potdd;

      calcpsidd2 = &calcpsidd2_cpy;

      cudaCheckFFTError(cufftSetStream(planFWR, exec_stream));
      cudaCheckFFTError(cufftSetStream(planFWC, exec_stream));
      cudaCheckFFTError(cufftSetStream(planBWC, exec_stream));
      cudaCheckFFTError(cufftSetStream(planBWR, exec_stream));

      break;
   case 2:
      d_dtensor1 = alloc_double_tensor_device(localNx, Ny, Nz);
      d_dtensor2 = alloc_double_tensor_device(Nx, localNy, Nz);
      d_pot = d_dtensor1;
      d_potdd = d_dtensor2;

      potparam.dstPtr = d_pot;
      potddparam.dstPtr = d_potdd;

      cudaCheckError(cudaMemcpy3D(&potparam));
      cudaCheckError(cudaMemcpy3D(&potddparam));

      calcpsidd2 = &calcpsidd2_mem;

      break;
   default:
      fprintf(stderr, "Invalid POTMEM value %d\n", potmem);
      MPI_Finalize(); exit(EXIT_FAILURE);
   }


   int dim = 2;
   long long nfr[] = {Ny, Nz};
   long long howmany = localNx;
   long long idist = Ny * (psidd2.pitch / sizeof(cufftDoubleReal)), odist = Ny * (psidd2.pitch / sizeof(cufftDoubleComplex));
   long long istride = 1, ostride = 1;
   long long inembedfr[] = {Ny, psidd2.pitch / sizeof(cufftDoubleReal)}, onembedfr[] = {Ny, psidd2.pitch / sizeof(cufftDoubleComplex)};

   cudaCheckFFTError(cufftMakePlanMany64(planFWR, dim, nfr, inembedfr, istride, idist, onembedfr, ostride, odist, CUFFT_D2Z, howmany, &workSizeFWR));


   dim = 1;
   long long nfc[] = {Nx};
   howmany = Nz2 + 1; // howmany = localNy * (Nz2 + 1);
   idist = 1; odist = 1;
   istride = localNy * psidd2_t.pitch / sizeof(cufftDoubleComplex); ostride = localNy * psidd2_t.pitch / sizeof(cufftDoubleComplex);
   long long inembedfc[] = {Nx}, onembedfc[] = {Nx};

   cudaCheckFFTError(cufftMakePlanMany64(planFWC, dim, nfc, inembedfc, istride, idist, onembedfc, ostride, odist, CUFFT_Z2Z, howmany, &workSizeFWC));

   long long nbc[] = {Nx};
   howmany = Nz2 + 1; // howmany = localNy * (Nz2 + 1);
   idist = 1; odist = 1;
   istride = localNy * psidd2_t.pitch / sizeof(cufftDoubleComplex); ostride = localNy * psidd2_t.pitch / sizeof(cufftDoubleComplex);
   long long inembedbc[] = {Nx}, onembedbc[] = {Nx};

   cudaCheckFFTError(cufftMakePlanMany64(planBWC, dim, nbc, inembedbc, istride, idist, onembedbc, ostride, odist, CUFFT_Z2Z, howmany, &workSizeBWC));

   dim = 2;
   long long nbr[] = {Ny, Nz};
   howmany = localNx;
   idist = Ny * (psidd2.pitch / sizeof(cufftDoubleComplex)); odist = Ny * psidd2.pitch / sizeof(cufftDoubleReal);
   istride = 1; ostride = 1;
   long long inembedbr[] = {Ny, psidd2.pitch / sizeof(cufftDoubleComplex)}, onembedbr[] = {Ny, psidd2.pitch / sizeof(cufftDoubleReal)};

   cudaCheckFFTError(cufftMakePlanMany64(planBWR, dim, nbr, inembedbr, istride, idist, onembedbr, ostride, odist, CUFFT_Z2D, howmany, &workSizeBWR));

   workSizeMax = 0;
   workSizeMax = MAX(workSizeMax, workSizeFWR, workSizeFWC);
   workSizeMax = MAX(workSizeMax, workSizeBWR, workSizeBWC);

   workArea = alloc_double_vector_device(workSizeMax / sizeof(double));

   cudaCheckFFTError(cufftSetWorkArea(planFWR, workArea));
   cudaCheckFFTError(cufftSetWorkArea(planFWC, workArea));
   cudaCheckFFTError(cufftSetWorkArea(planBWC, workArea));
   cudaCheckFFTError(cufftSetWorkArea(planBWR, workArea));


   cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

   dimBlock3d.x = dimBlock3d.y = dimBlock3d.z = CUDA_BLOCK_SIZE_3D;
   dimGrid3d.x = dimGrid3d.y = dimGrid3d.z = numSMs;

   dimBlock2d.x = dimBlock2d.y = CUDA_BLOCK_SIZE_2D;
   dimGrid2d.x = dimGrid2d.y = numSMs;

   dimBlock1d.x = CUDA_BLOCK_SIZE;
   dimGrid1d.x = numSMs;

   tran_psi = init_transpose_double(nprocs, localNx, localNy, Ny, Nz, psi.ptr, psi.pitch, psi_t.ptr, psi_t.pitch);
   tran_dpsi = init_transpose_double(nprocs, localNx, localNy, Ny, Nz, dpsi.ptr, dpsi.pitch, dpsi_t.ptr, dpsi_t.pitch);
   tran_dd2 = init_transpose_complex(nprocs, localNx, localNy, Ny, Nz2 + 1, psidd2.ptr, psidd2.pitch, psidd2_t.ptr, psidd2_t.pitch);
}

/**
 *    Calculation of psi^2.
 *    psi  - array with the wave function values
 *    psi2 - array with the squared wave function values
 */
void calcpsi2(cudaPitchedPtr psi, cudaPitchedPtr psi2) {

   calcpsi2_kernel<<<dimGrid3d, dimBlock3d>>>(psi, psi2);

   return;
}

__global__ void calcpsi2_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2) {
   long cnti, cntj, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psi2row = get_double_tensor_row(psi2, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            tmp = psirow[cntk];
            psi2row[cntk] = tmp * tmp;
         }
      }
   }
}

/**
 *    Calculation of the wave function norm and normalization.
 *    norm   - wave function norm
 *    psi    - array with the wave function values
 *    tmpxyz - temporary array
 *    tmpxy  - temporary array
 *    tmpx   - temporary array
 */
void calcnorm(double *norm, cudaPitchedPtr psi, cudaPitchedPtr tmpxyz, cudaPitchedPtr tmpxy, double *tmpx) {
   double tmp;
   void *sendbuf;

   calcpsi2(psi, tmpxyz);

   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dz, tmpxyz, tmpxy, localNx, Ny, Nz);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dy, tmpxy, tmpx, localNx, Ny);

   cudaCheckError(cudaDeviceSynchronize());

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      *norm = sqrt(simpint_gpu(dx, tmpx, Nx));
   }

   MPI_Bcast(norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   tmp = 1. / *norm;

   calcnorm_kernel<<<dimGrid3d, dimBlock3d>>>(psi, tmp);

   return;
}

__global__ void calcnorm_kernel(cudaPitchedPtr psi, double tmp) {
   long cnti, cntj, cntk;
   double *psirow;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psirow[cntk] *= tmp;
         }
      }
   }
}

/**
 *    Calculation of the chemical potential and energy.
 *    mu      - chemical potential
 *    en      - energy
 *    psi     - array with the wave function values
 *    psi_t   - array with the transposed wave function values
 *    dpsi    - array with the wave function derivatives
 *    dpsi_t  - array with the transposed wave function derivatives
 *    psidd2  - array with the squared wave function values
 *    tmpxyz  - temporary array
 *    tmpmuen - temporary array
 *    tmpxy   - temporary array
 *    tmpy   - temporary array
 *    tmpx    - temporary array
 */
void calcmuen(double *mu, double *en, cudaPitchedPtr psi, cudaPitchedPtr psi_t, cudaPitchedPtr dpsi, cudaPitchedPtr dpsi_t, cudaPitchedPtr psidd2, cudaPitchedPtr psidd2_t, cudaPitchedPtr psidd2tmp, cudaPitchedPtr tmpmu, cudaPitchedPtr tmpen, cudaPitchedPtr tmpxy, double *tmpx) {
   void *sendbuf;

   calcpsidd2(psi, psi_t, psidd2, psidd2_t, psidd2tmp);

   transpose(tran_psi);

   calcmuen_kernel1x<<<dimGrid3d, dimBlock3d>>>(psi_t, dpsi_t, par);

   transpose_back(tran_dpsi);

   calcmuen_kernel1y<<<dimGrid3d, dimBlock3d>>>(psi, dpsi, par);
   calcmuen_kernel1z<<<dimGrid3d, dimBlock3d>>>(psi, dpsi, par);

   calcmuen_kernel2<<<dimGrid3d, dimBlock3d>>>(psi, dpsi, psidd2, d_pot, tmpmu, tmpen, g, gd);

   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dz, tmpmu, tmpxy, localNx, Ny, Nz);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dy, tmpxy, tmpx, localNx, Ny);

   cudaCheckError(cudaDeviceSynchronize());

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      *mu = simpint_gpu(dx, tmpx, Nx);
   }

   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dz, tmpen, tmpxy, localNx, Ny, Nz);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dy, tmpxy, tmpx, localNx, Ny);

   cudaCheckError(cudaDeviceSynchronize());

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      *en = simpint_gpu(dx, tmpx, Nx);
   }

   return;
}

__global__ void calcmuen_kernel1x(cudaPitchedPtr psi, cudaPitchedPtr dpsi, double par) {
   long cnti, cntj, cntk;
   double *psirow1, *psirow2, *psirow3, *psirow4, *dpsirow;
   double dpsix;

   for (cntj = TID_Z; cntj < d_localNy; cntj += GRID_STRIDE_Z) {
      for (cntk = TID_Y; cntk < d_Nz; cntk += GRID_STRIDE_Y) {
         for (cnti = TID_X; cnti < d_Nx; cnti += GRID_STRIDE_X) {
            dpsirow = get_double_tensor_row(dpsi, cnti, cntj);

            if (cnti == 0) {
               dpsix = 0.;
            }

            if (cnti == 1) {
               psirow1 = get_double_tensor_row(psi, 2, cntj);
               psirow2 = get_double_tensor_row(psi, 0, cntj);

               dpsix = (psirow1[cntk] - psirow2[cntk]) / (2. * d_dx);
            }

            if (cnti > 1 && cnti < d_Nx - 2) {
               psirow1 = get_double_tensor_row(psi, cnti - 2, cntj);
               psirow2 = get_double_tensor_row(psi, cnti - 1, cntj);
               psirow3 = get_double_tensor_row(psi, cnti + 1, cntj);
               psirow4 = get_double_tensor_row(psi, cnti + 2, cntj);

               dpsix = (psirow1[cntk] - 8. * psirow2[cntk] + 8. * psirow3[cntk] - psirow4[cntk]) / (12. * d_dx);
            }

            if (cnti == d_Nx - 2) {
               psirow1 = get_double_tensor_row(psi, d_Nx - 1, cntj);
               psirow2 = get_double_tensor_row(psi, d_Nx - 3, cntj);

               dpsix = (psirow1[cntk] - psirow2[cntk]) / (2. * d_dx);
            }

            if (cnti == d_Nx - 1) {
               dpsix = 0.;
            }

            dpsirow[cntk] = (dpsix * dpsix) / (3. - par);
         }
      }
   }
}

__global__ void calcmuen_kernel1y(cudaPitchedPtr psi, cudaPitchedPtr dpsi, double par) {
   long cnti, cntj, cntk;
   double *psirow1, *psirow2, *psirow3, *psirow4, *dpsirow;
   double dpsiy;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntk = TID_Y; cntk < d_Nz; cntk += GRID_STRIDE_Y) {
         for (cntj = TID_X; cntj < d_Ny; cntj += GRID_STRIDE_X) {
            dpsirow = get_double_tensor_row(dpsi, cnti, cntj);

            if (cntj == 0) {
               dpsiy = 0.;
            }

            if (cntj == 1) {
               psirow1 = get_double_tensor_row(psi, cnti, 2);
               psirow2 = get_double_tensor_row(psi, cnti, 0);

               dpsiy = (psirow1[cntk] - psirow2[cntk]) / (2. * d_dy);
            }

            if (cntj > 1 && cntj < d_Ny - 2) {
               psirow1 = get_double_tensor_row(psi, cnti, cntj - 2);
               psirow2 = get_double_tensor_row(psi, cnti, cntj - 1);
               psirow3 = get_double_tensor_row(psi, cnti, cntj + 1);
               psirow4 = get_double_tensor_row(psi, cnti, cntj + 2);

               dpsiy = (psirow1[cntk] - 8. * psirow2[cntk] + 8. * psirow3[cntk] - psirow4[cntk]) / (12. * d_dy);
            }

            if (cntj == d_Ny - 2) {
               psirow1 = get_double_tensor_row(psi, cnti, d_Ny - 1);
               psirow2 = get_double_tensor_row(psi, cnti, d_Ny - 3);

               dpsiy = (psirow1[cntk] - psirow2[cntk]) / (2. * d_dy);
            }

            if (cntj == d_Ny - 1) {
               dpsiy = 0.;
            }

            dpsirow[cntk] += (dpsiy * dpsiy) / (3. - par);
         }
      }
   }
}

__global__ void calcmuen_kernel1z(cudaPitchedPtr psi, cudaPitchedPtr dpsi, double par) {
   long cnti, cntj, cntk;
   double *psirow, *dpsirow;
   double dpsiz;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         dpsirow = get_double_tensor_row(dpsi, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            if (cntk == 0) {
               dpsiz = 0.;
            }

            if (cntk == 1) {
               dpsiz = (psirow[2] - psirow[0]) / (2. * d_dz);
            }

            if (cntk > 1 && cntk < d_Nz - 2) {
               dpsiz = (psirow[cntk - 2] - 8. * psirow[cntk - 1] + 8. * psirow[cntk + 1] - psirow[cntk + 2]) / (12. * d_dz);
            }

            if (cntk == d_Nz - 2) {
               dpsiz = (psirow[d_Nz - 1] - psirow[d_Nz - 3]) / (2. * d_dz);
            }

            if (cntk == d_Nz - 1) {
               dpsiz = 0.;
            }

            dpsirow[cntk] += (dpsiz * dpsiz) / (3. - par);
         }
      }
   }
}

__global__ void calcmuen_kernel2(cudaPitchedPtr psi, cudaPitchedPtr dpsi, cudaPitchedPtr psidd2, cudaPitchedPtr pot, cudaPitchedPtr tmpmu, cudaPitchedPtr tmpen, double g, double gd) {
   long cnti, cntj, cntk;
   double *psirow, *psidd2row, *tmpmurow, *tmpenrow, *potrow, *dpsirow;
   double psi2, psi2lin, psidd2lin;
   double tmppot;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psidd2row = get_double_tensor_row(psidd2, cnti, cntj);
         potrow = get_double_tensor_row(pot, cnti, cntj);
         tmpmurow = get_double_tensor_row(tmpmu, cnti, cntj);
         tmpenrow = get_double_tensor_row(tmpen, cnti, cntj);
         dpsirow = get_double_tensor_row(dpsi, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psi2 = psirow[cntk];
            psi2 *= psi2;
            psi2lin = psi2 * g;
            psidd2lin = (psidd2row[cntk] / (d_Nx * d_Ny * d_Nz)) * gd;
            tmppot = potrow[cntk];
            tmpmurow[cntk] = (tmppot + psi2lin + psidd2lin) * psi2 + dpsirow[cntk];
            tmpenrow[cntk] = (tmppot + 0.5 * psi2lin + 0.5 * psidd2lin) * psi2 + dpsirow[cntk];
         }
      }
   }
}

/**
 *    Calculation of squared wave function values for dipole-dipole
 *    interaction.
 *    psi      - array with the wave function values
 *    psi_t    - temporary array holding transposed wave function
 *    psidd2   - array with the squared wave function values (scaled by Nx * Ny * Nz)
 *    psidd2_t - temporary array holding transposed FFT values
 */
void calcpsidd2_mem(cudaPitchedPtr psi, cudaPitchedPtr psi_t, cudaPitchedPtr psidd2, cudaPitchedPtr psidd2_t, cudaPitchedPtr psidd2tmp) {
   long cnti;
   long last = 0;
   cufftDoubleComplex *psidd2ptr;

   calcpsi2(psi, psidd2);

   cudaCheckFFTError(cufftExecD2Z(planFWR, (cufftDoubleReal *) psidd2.ptr, (cufftDoubleComplex *) psidd2.ptr));

   transpose(tran_dd2);

   //cudaCheckFFTError(cufftExecZ2Z(planFWC, (cufftDoubleComplex *) psidd2_t.ptr, (cufftDoubleComplex *) psidd2_t.ptr, CUFFT_FORWARD));
   psidd2ptr = (cufftDoubleComplex *) psidd2_t.ptr;
   for (cnti = 0; cnti < localNy; cnti ++) {
      cudaCheckFFTError(cufftExecZ2Z(planFWC, psidd2ptr, psidd2ptr, CUFFT_FORWARD));
      psidd2ptr += (psidd2_t.pitch / sizeof(cufftDoubleComplex));
   }

   calcpsidd2_kernel1<<<dimGrid3d, dimBlock3d>>>(psidd2_t, d_potdd);

   //cudaCheckFFTError(cufftExecZ2Z(planBWC, (cufftDoubleComplex *) psidd2_t.ptr, (cufftDoubleComplex *) psidd2_t.ptr, CUFFT_INVERSE));
   psidd2ptr = (cufftDoubleComplex *) psidd2_t.ptr;
   for (cnti = 0; cnti < localNy; cnti ++) {
      cudaCheckFFTError(cufftExecZ2Z(planBWC, psidd2ptr, psidd2ptr, CUFFT_INVERSE));
      psidd2ptr += (psidd2_t.pitch / sizeof(cufftDoubleComplex));
   }

   transpose_back(tran_dd2);

   cudaCheckFFTError(cufftExecZ2D(planBWR, (cufftDoubleComplex *) psidd2.ptr, (cufftDoubleReal *) psidd2.ptr));

   if (nprocs > 1) {
      if (rank == 0) {
         MPI_Send(psidd2.ptr, Ny * (psidd2.pitch / sizeof(double)), MPI_DOUBLE, nprocs - 1, 0, MPI_COMM_WORLD);
      } else if (rank == nprocs - 1) {
         MPI_Recv(psidd2tmp.ptr, Ny * (psidd2tmp.pitch / sizeof(double)), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         last = 1;
         calcpsidd2_kernel2a<<<dimGrid2d, dimBlock2d>>>(psidd2, psidd2tmp);
      }
   } else {
      cudaCheckError(cudaMemcpy(psidd2tmp.ptr, psidd2.ptr, Ny * psidd2.pitch, cudaMemcpyDeviceToDevice));
      last = 1;
      calcpsidd2_kernel2a<<<dimGrid2d, dimBlock2d>>>(psidd2, psidd2tmp);
   }

   calcpsidd2_kernel2b<<<dimGrid2d, dimBlock2d>>>(psidd2, last);

   cudaCheckError(cudaDeviceSynchronize());

   return;
}

void calcpsidd2_cpy(cudaPitchedPtr psi, cudaPitchedPtr psi_t, cudaPitchedPtr psidd2, cudaPitchedPtr psidd2_t, cudaPitchedPtr psidd2tmp) {
   long cnti;
   long last = 0;
   cufftDoubleComplex *psidd2ptr;

   cudaCheckError(cudaMemcpy3DAsync(&potddparam, data_stream));

   calcpsi2(psi, psidd2);

   cudaCheckFFTError(cufftExecD2Z(planFWR, (cufftDoubleReal *) psidd2.ptr, (cufftDoubleComplex *) psidd2.ptr));
   cudaCheckError(cudaStreamSynchronize(exec_stream));

   transpose(tran_dd2);

   //cudaCheckFFTError(cufftExecZ2Z(planFWC, (cufftDoubleComplex *) psidd2_t.ptr, (cufftDoubleComplex *) psidd2_t.ptr, CUFFT_FORWARD));
   psidd2ptr = (cufftDoubleComplex *) psidd2_t.ptr;
   for (cnti = 0; cnti < localNy; cnti ++) {
      cudaCheckFFTError(cufftExecZ2Z(planFWC, psidd2ptr, psidd2ptr, CUFFT_FORWARD));
      psidd2ptr += (psidd2_t.pitch / sizeof(cufftDoubleComplex));
   }

   cudaCheckError(cudaStreamSynchronize(data_stream));

   calcpsidd2_kernel1<<<dimGrid3d, dimBlock3d, 0, exec_stream>>>(psidd2_t, d_potdd);

   cudaCheckError(cudaStreamSynchronize(exec_stream));
   cudaCheckError(cudaMemcpy3DAsync(&potparam, data_stream));

   //cudaCheckFFTError(cufftExecZ2Z(planBWC, (cufftDoubleComplex *) psidd2_t.ptr, (cufftDoubleComplex *) psidd2_t.ptr, CUFFT_INVERSE));
   psidd2ptr = (cufftDoubleComplex *) psidd2_t.ptr;
   for (cnti = 0; cnti < localNy; cnti ++) {
      cudaCheckFFTError(cufftExecZ2Z(planBWC, psidd2ptr, psidd2ptr, CUFFT_INVERSE));
      psidd2ptr += (psidd2_t.pitch / sizeof(cufftDoubleComplex));
   }
   cudaCheckError(cudaStreamSynchronize(exec_stream));

   transpose_back(tran_dd2);

   cudaCheckFFTError(cufftExecZ2D(planBWR, (cufftDoubleComplex *) psidd2.ptr, (cufftDoubleReal *) psidd2.ptr));

   if (nprocs > 1) {
      if (rank == 0) {
         MPI_Send(psidd2.ptr, Ny * (psidd2.pitch / sizeof(double)), MPI_DOUBLE, nprocs - 1, 0, MPI_COMM_WORLD);
      } else if (rank == nprocs - 1) {
         MPI_Recv(psidd2tmp.ptr, Ny * (psidd2tmp.pitch / sizeof(double)), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         last = 1;
         calcpsidd2_kernel2a<<<dimGrid2d, dimBlock2d>>>(psidd2, psidd2tmp);
      }
   } else {
      cudaCheckError(cudaMemcpy(psidd2tmp.ptr, psidd2.ptr, Ny * psidd2.pitch, cudaMemcpyDeviceToDevice));
      last = 1;
      calcpsidd2_kernel2a<<<dimGrid2d, dimBlock2d>>>(psidd2, psidd2tmp);
   }

   calcpsidd2_kernel2b<<<dimGrid2d, dimBlock2d, 0, exec_stream>>>(psidd2, last);

   return;
}

__global__ void calcpsidd2_kernel1(cudaPitchedPtr psidd2, cudaPitchedPtr potdd) {
   long cnti, cntj, cntk;
   double *potddrow;
   cuDoubleComplex *psidd2row;
   double tmp;

   for (cnti = TID_Z; cnti < d_Nx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_localNy; cntj += GRID_STRIDE_Y) {
         psidd2row = get_complex_tensor_row(psidd2, cnti, cntj);
         potddrow = get_double_tensor_row(potdd, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz / 2 + 1; cntk += GRID_STRIDE_X) {
            tmp = potddrow[cntk];
            psidd2row[cntk].x *= tmp;
            psidd2row[cntk].y *= tmp;
         }
      }
   }
}

__global__ void calcpsidd2_kernel2a(cudaPitchedPtr psidd2, cudaPitchedPtr tmpyz) {
   long cntj, cntk;
   double *psidd2firstrow, *psidd2lastrow;

   for (cntj = TID_Y; cntj < d_Ny - 1; cntj += GRID_STRIDE_Y) {
      psidd2firstrow = get_double_matrix_row(tmpyz, cntj);
      psidd2lastrow = get_double_tensor_row(psidd2, d_localNx - 1, cntj);

      for (cntk = TID_X; cntk < d_Nz - 1; cntk += GRID_STRIDE_X) {
         psidd2lastrow[cntk] = psidd2firstrow[cntk];
      }
   }
}

__global__ void calcpsidd2_kernel2b(cudaPitchedPtr psidd2, long last) {
   long cnti, cntj, cntk;
   double *psidd2row, *psidd2firstrow, *psidd2lastrow;

   for (cnti = TID_Y; cnti < d_localNx - last; cnti += GRID_STRIDE_Y) {
      psidd2firstrow = get_double_tensor_row(psidd2, cnti, 0);
      psidd2lastrow = get_double_tensor_row(psidd2, cnti, d_Ny - 1);

      for (cntk = TID_X; cntk < d_Nz - 1; cntk += GRID_STRIDE_X) {
         psidd2lastrow[cntk] = psidd2firstrow[cntk];
      }
   }

   for (cnti = TID_Y; cnti < d_localNx - last; cnti += GRID_STRIDE_Y) {
      for (cntj = TID_X; cntj < d_Ny - 1; cntj += GRID_STRIDE_X) {
         psidd2row = get_double_tensor_row(psidd2, cnti, cntj);

         psidd2row[d_Nz - 1] = psidd2row[0];
      }
   }
}

/**
 *    Calculation of the root mean square radius.
 *    rms    - root mean square radius
 *    psi    - array with the wave function values
 *    psi_t  - temporary array to hold transposed wave function
 *    tmpxyz - temporary array
 *    tmpyz  - temporary array
 *    tmpxz  - temporary array
 *    tmpxy  - temporary array
 *    tmpx   - temporary array
 *    tmpy   - temporary array
 */
void calcrms(double *rms, cudaPitchedPtr psi, cudaPitchedPtr psi_t, cudaPitchedPtr tmpyzx, cudaPitchedPtr tmpxzy, cudaPitchedPtr tmpxyz, cudaPitchedPtr tmpyz, cudaPitchedPtr tmpxz, cudaPitchedPtr tmpxy, double *tmpx, double *tmpy) {
   void *sendbuf;

   transpose(tran_psi);

   calcrms_kernel1x<<<dimGrid3d, dimBlock3d>>>(psi_t, tmpyzx, d_x2);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dx, tmpyzx, tmpyz, localNy, Nz, Nx);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dz, tmpyz, tmpy, localNy, Nz);

   cudaCheckError(cudaDeviceSynchronize());

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpy;
   MPI_Gather(sendbuf, localNy, MPI_DOUBLE, tmpy, localNy, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[1] = sqrt(simpint_gpu(dy, tmpy, Ny));
   }

   calcrms_kernel1y<<<dimGrid3d, dimBlock3d>>>(psi, tmpxzy, d_y2);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dx, tmpxzy, tmpxz, localNx, Nz, Ny);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dz, tmpxz, tmpx, localNx, Nz);

   cudaCheckError(cudaDeviceSynchronize());

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[2] = sqrt(simpint_gpu(dx, tmpx, Nx));
   }

   calcrms_kernel1z<<<dimGrid3d, dimBlock3d>>>(psi, tmpxyz, d_z2);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dz, tmpxyz, tmpxy, localNx, Ny, Nz);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dy, tmpxy, tmpx, localNx, Ny);

   cudaCheckError(cudaDeviceSynchronize());

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[3] = sqrt(simpint_gpu(dx, tmpx, Nx));
      rms[0] = sqrt(rms[1] * rms[1] + rms[2] * rms[2] + rms[3] * rms[3]);
   }

   return;
}

__global__ void calcrms_kernel1x(cudaPitchedPtr psi, cudaPitchedPtr tmpyzx, double *x2) {
   long cnti, cntj, cntk;
   double *psirow, *tmpyzxrow;
   double psi2;

   for (cntj = TID_Z; cntj < d_localNy; cntj += GRID_STRIDE_Z) {
      for (cntk = TID_Y; cntk < d_Nz; cntk += GRID_STRIDE_Y) {
         tmpyzxrow = get_double_tensor_row(tmpyzx, cntj, cntk);

         for (cnti = TID_X; cnti < d_Nx; cnti += GRID_STRIDE_X) {
            psirow = get_double_tensor_row(psi, cnti, cntj);

            psi2 = psirow[cntk];
            psi2 *= psi2;
            tmpyzxrow[cnti] = x2[cnti] * psi2;
         }
      }
   }
}

__global__ void calcrms_kernel1y(cudaPitchedPtr psi, cudaPitchedPtr tmpxzy, double *y2) {
   long cnti, cntj, cntk;
   double *psirow, *tmpxzyrow;
   double psi2;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntk = TID_Y; cntk < d_Nz; cntk += GRID_STRIDE_Y) {
         tmpxzyrow = get_double_tensor_row(tmpxzy, cnti, cntk);

         for (cntj = TID_X; cntj < d_Ny; cntj += GRID_STRIDE_X) {
            psirow = get_double_tensor_row(psi, cnti, cntj);

            psi2 = psirow[cntk];
            psi2 *= psi2;
            tmpxzyrow[cntj] = y2[cntj] * psi2;
         }
      }
   }
}

__global__ void calcrms_kernel1z(cudaPitchedPtr psi, cudaPitchedPtr tmpxyz, double *z2) {
   long cnti, cntj, cntk;
   double *psirow, *tmpxyzrow;
   double psi2;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         tmpxyzrow = get_double_tensor_row(tmpxyz, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psi2 = psirow[cntk];
            psi2 *= psi2;
            tmpxyzrow[cntk] = z2[cntk] * psi2;
         }
      }
   }
}

/**
 *    Time propagation with respect to H1 (part of the Hamiltonian without
 *    spatial derivatives).
 *    psi    - array with the wave function values
 *    psidd2 - array with the squared wave function values
 */
void calcnu(cudaPitchedPtr psi, cudaPitchedPtr psidd2) {

   calcnu_kernel<<<dimGrid3d, dimBlock3d>>>(psi, psidd2, d_pot, g, gd);

   return;
}

__global__ void calcnu_kernel(cudaPitchedPtr psi, cudaPitchedPtr psidd2, cudaPitchedPtr pot, double g, double gd) {
   long cnti, cntj, cntk;
   double psi2lin, psidd2lin, psitmp, pottmp;
   double *psirow, *psidd2row, *potrow;

   for (cnti = TID_Z; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psidd2row = get_double_tensor_row(psidd2, cnti, cntj);
         potrow = get_double_tensor_row(pot, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psitmp = psirow[cntk];
            psidd2lin = psidd2row[cntk];
            pottmp = potrow[cntk];

            psi2lin = psitmp * psitmp * g;
            psidd2lin = (psidd2lin / (d_Nx * d_Ny * d_Nz)) * gd;
            pottmp = d_dt * (pottmp + psi2lin + psidd2lin);

            psirow[cntk] = psitmp * exp(-pottmp);
         }
      }
   }
}

/**
 *    Time propagation with respect to H2 (x-part of the Laplacian).
 *    psi_t - temporary array holding transposed wave function
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calclux(cudaPitchedPtr psi_t, cudaPitchedPtr cbeta) {

   transpose(tran_psi);

   calclux_kernel<<<dimGrid2d, dimBlock2d>>>(psi_t, cbeta, d_calphax, d_cgammax);
   cudaCheckError(cudaDeviceSynchronize());

   transpose_back(tran_psi);

   return;
}

__global__ void calclux_kernel(cudaPitchedPtr psi, cudaPitchedPtr cbeta, double *calphax, double *cgammax) {
   long cnti, cntj, cntk;
   double c;
   double *psirowprev, *psirowcurr, *psirownext, *cbetarowprev, *cbetarowcurr;

   for (cntj = TID_Y; cntj < d_localNy; cntj += GRID_STRIDE_Y) {
      for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
         cbetarowcurr = get_double_tensor_row(cbeta, d_Nx - 2, cntj);
         psirowcurr = get_double_tensor_row(psi, d_Nx - 1, cntj);

         cbetarowcurr[cntk] = psirowcurr[cntk];

         for(cnti = d_Nx - 2; cnti > 0; cnti --) {
            cbetarowprev = get_double_tensor_row(cbeta, cnti - 1, cntj);
            cbetarowcurr = get_double_tensor_row(cbeta, cnti, cntj);

            psirowprev = get_double_tensor_row(psi, cnti - 1, cntj);
            psirowcurr = get_double_tensor_row(psi, cnti, cntj);
            psirownext = get_double_tensor_row(psi, cnti + 1, cntj);

            c = - d_Ax * psirownext[cntk] + d_Ax0r * psirowcurr[cntk] - d_Ax * psirowprev[cntk];
            cbetarowprev[cntk] = cgammax[cnti] * (d_Ax * cbetarowcurr[cntk] - c);
         }

         psirowcurr = get_double_tensor_row(psi, 0, cntj);

         psirowcurr[cntk] = 0.;

         for(cnti = 0; cnti < d_Nx - 2; cnti ++) {
            cbetarowcurr = get_double_tensor_row(cbeta, cnti, cntj);

            psirowcurr = get_double_tensor_row(psi, cnti, cntj);
            psirownext = get_double_tensor_row(psi, cnti + 1, cntj);

            psirownext[cntk] = fma(calphax[cnti], psirowcurr[cntk], cbetarowcurr[cntk]);
         }

         psirowcurr = get_double_tensor_row(psi, d_Nx - 1, cntj);

         psirowcurr[cntk] = 0.;
      }
   }
}

/**
 *    Time propagation with respect to H3 (y-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calcluy(cudaPitchedPtr psi, cudaPitchedPtr cbeta) {

   calcluy_kernel<<<dimGrid2d, dimBlock2d>>>(psi, cbeta, d_calphay, d_cgammay);

   return;
}

__global__ void calcluy_kernel(cudaPitchedPtr psi, cudaPitchedPtr cbeta, double *calphay, double *cgammay) {
   long cnti, cntj, cntk;
   double c;
   double *psirowprev, *psirowcurr, *psirownext, *cbetarowprev, *cbetarowcurr;

   for (cnti = TID_Y; cnti < d_localNx; cnti += GRID_STRIDE_Y) {
      for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
         cbetarowcurr = get_double_tensor_row(cbeta, cnti, d_Ny - 2);
         psirowcurr = get_double_tensor_row(psi, cnti, d_Ny - 1);

         cbetarowcurr[cntk] = psirowcurr[cntk];

         for(cntj = d_Ny - 2; cntj > 0; cntj --) {
            cbetarowprev = get_double_tensor_row(cbeta, cnti, cntj - 1);
            cbetarowcurr = get_double_tensor_row(cbeta, cnti, cntj);

            psirowprev = get_double_tensor_row(psi, cnti, cntj - 1);
            psirowcurr = get_double_tensor_row(psi, cnti, cntj);
            psirownext = get_double_tensor_row(psi, cnti, cntj + 1);

            c = - d_Ay * psirownext[cntk] + d_Ay0r * psirowcurr[cntk] - d_Ay * psirowprev[cntk];
            cbetarowprev[cntk] = cgammay[cntj] * (d_Ay * cbetarowcurr[cntk] - c);
         }

         psirowcurr = get_double_tensor_row(psi, cnti, 0);

         psirowcurr[cntk] = 0.;

         for(cntj = 0; cntj < d_Ny - 2; cntj ++) {
            cbetarowcurr = get_double_tensor_row(cbeta, cnti, cntj);

            psirowcurr = get_double_tensor_row(psi, cnti, cntj);
            psirownext = get_double_tensor_row(psi, cnti, cntj + 1);

            psirownext[cntk] = fma(calphay[cntj], psirowcurr[cntk], cbetarowcurr[cntk]);
         }

         psirowcurr = get_double_tensor_row(psi, cnti, d_Ny - 1);

         psirowcurr[cntk] = 0.;
      }
   }
}

/**
 *    Time propagation with respect to H4 (z-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
void calcluz(cudaPitchedPtr psi, cudaPitchedPtr cbeta) {

   calcluz_kernel<<<dimGrid2d, dimBlock2d>>>(psi, cbeta, d_calphaz, d_cgammaz);

   return;
}

__global__ void calcluz_kernel(cudaPitchedPtr psi, cudaPitchedPtr cbeta, double *calphaz, double *cgammaz) {
   long cnti, cntj, cntk;
   double c;
   double *psirow, *cbetarow;

   for (cnti = TID_Y; cnti < d_localNx; cnti += GRID_STRIDE_Y) {
      for (cntj = TID_X; cntj < d_Ny; cntj += GRID_STRIDE_X) {
         cbetarow = get_double_tensor_row(cbeta, cnti, cntj);
         psirow = get_double_tensor_row(psi, cnti, cntj);

         cbetarow[d_Nz - 2] = psirow[d_Nz - 1];

         for(cntk = d_Nz - 2; cntk > 0; cntk --) {
            c = - d_Az * psirow[cntk + 1] + d_Az0r * psirow[cntk] - d_Az * psirow[cntk - 1];
            cbetarow[cntk - 1] = cgammaz[cntk] * (d_Az * cbetarow[cntk] - c);
         }

         psirow[0] = 0.;

         for(cntk = 0; cntk < d_Nz - 2; cntk ++) {
            psirow[cntk + 1] = fma(calphaz[cntk], psirow[cntk], cbetarow[cntk]);
         }

         psirow[d_Nz - 1] = 0.;
      }
   }

   return;
}

void outdenx(cudaPitchedPtr psi, cudaPitchedPtr tmpxyz, cudaPitchedPtr tmpxy, double *tmpx, double *h_tmpx, MPI_File file) {
   long cnti;
   MPI_Offset fileoffset;
   double tmp[2];

   outdenx_kernel<<<dimGrid3d, dimBlock3d>>>(psi, tmpxyz, outstpx);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dz, tmpxyz, tmpxy, localNx / outstpx, Ny, Nz);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dy, tmpxy, tmpx, localNx / outstpx, Ny);

   cudaCheckError(cudaMemcpy(h_tmpx, tmpx, (localNx / outstpx) * sizeof(double), cudaMemcpyDeviceToHost));

   fileoffset = rank * 2 * sizeof(double) * (localNx / outstpx);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      tmp[0] = x[offsetNx + cnti];
      tmp[1] = h_tmpx[cnti / outstpx];
      MPI_File_write_at_all(file, fileoffset, &tmp, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
      fileoffset += 2 * sizeof(double);
   }
}

__global__ void outdenx_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpx) {
   long cnti, cntj, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Z * outstpx; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psi2row = get_double_tensor_row(psi2, cnti / outstpx, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            tmp = psirow[cntk];
            psi2row[cntk] = tmp * tmp;
         }
      }
   }
}

void outdeny(cudaPitchedPtr psi_t, cudaPitchedPtr tmpyxz, cudaPitchedPtr tmpyx, double *tmpy, double *h_tmpy, MPI_File file) {
   long cntj;
   MPI_Offset fileoffset;
   double tmp[2];

   transpose(tran_psi);

   outdeny_kernel<<<dimGrid3d, dimBlock3d>>>(psi_t, tmpyxz, outstpy);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dz, tmpyxz, tmpyx, localNy / outstpy, Nx, Nz);
   simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dx, tmpyx, tmpy, localNy / outstpy, Nx);

   cudaCheckError(cudaMemcpy(h_tmpy, tmpy, (localNy / outstpy) * sizeof(double), cudaMemcpyDeviceToHost));

   fileoffset = rank * 2 * sizeof(double) * (localNy / outstpy);

   for (cntj = 0; cntj < localNy; cntj += outstpy) {
      tmp[0] = y[offsetNy + cntj];
      tmp[1] = h_tmpy[cntj / outstpy];
      MPI_File_write_at_all(file, fileoffset, &tmp, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
      fileoffset += 2 * sizeof(double);
   }
}

__global__ void outdeny_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpy) {
   long cnti, cntj, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Z; cnti < d_Nx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y * outstpy; cntj < d_localNy; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psi2row = get_double_tensor_row(psi2, cntj / outstpy, cnti);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            tmp = psirow[cntk];
            psi2row[cntk] = tmp * tmp;
         }
      }
   }
}

void outdenz(cudaPitchedPtr psi, cudaPitchedPtr tmpxy, double *tmpx, double *h_tmpz, MPI_File file) {
   long cntk;
   MPI_Offset fileoffset;
   double tmp[2];
   void *sendbuf;

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;

   for (cntk = 0; cntk < Nz; cntk += outstpz) {
      outdenz_kernel<<<dimGrid2d, dimBlock2d>>>(psi, tmpxy, cntk);
      simpint2d_kernel<<<dimGrid1d, dimBlock1d>>>(dy, tmpxy, tmpx, localNx, Ny);
      cudaCheckError(cudaDeviceSynchronize());

      MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      if (rank == 0) {
         h_tmpz[cntk] = simpint_gpu(dx, tmpx, Nx);
      }
   }

   if (rank == 0) {
      fileoffset = 0;

      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         tmp[0] = z[cntk];
         tmp[1] = h_tmpz[cntk];
         MPI_File_write_at(file, fileoffset, &tmp, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
         fileoffset += 2 * sizeof(double);
      }
   }
}

__global__ void outdenz_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long indexNz) {
   long cnti, cntj;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Y; cnti < d_localNx; cnti += GRID_STRIDE_Y) {
      for (cntj = TID_X; cntj < d_Ny; cntj += GRID_STRIDE_X) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psi2row = get_double_matrix_row(psi2, cnti);

         tmp = psirow[indexNz];
         psi2row[cntj] = tmp * tmp;
      }
   }
}

void outdenxy(cudaPitchedPtr psi, cudaPitchedPtr tmpxyz, cudaPitchedPtr tmpxy, double **h_tmpxy, MPI_File file) {
   long cnti, cntj;
   cudaMemcpy3DParms dencpy = { 0 };
   MPI_Offset fileoffset;
   double tmp[3];

   outdenxy_kernel<<<dimGrid3d, dimBlock3d>>>(psi, tmpxyz, outstpx, outstpy);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dz, tmpxyz, tmpxy, localNx / outstpx, Ny / outstpy, Nz);

   dencpy.srcPtr = tmpxy;
   dencpy.dstPtr = make_cudaPitchedPtr(h_tmpxy[0], Ny * sizeof(double), Ny, localNx);
   dencpy.extent = make_cudaExtent(Ny * sizeof(double) / outstpy, localNx / outstpx, 1);
   dencpy.kind = cudaMemcpyDeviceToHost;
   cudaCheckError(cudaMemcpy3D(&dencpy));

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Ny / outstpy);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         tmp[0] = x[offsetNx + cnti];
         tmp[1] = y[cntj];
         tmp[2] = h_tmpxy[cnti / outstpx][cntj / outstpy];
         MPI_File_write_at_all(file, fileoffset, &tmp, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
         fileoffset += 3 * sizeof(double);
      }
   }
}

__global__ void outdenxy_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpx, long outstpy) {
   long cnti, cntj, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Z * outstpx; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y * outstpy; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psi2row = get_double_tensor_row(psi2, cnti / outstpx, cntj / outstpy);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            tmp = psirow[cntk];
            psi2row[cntk] = tmp * tmp;
         }
      }
   }
}

void outdenxz(cudaPitchedPtr psi, cudaPitchedPtr tmpxzy, cudaPitchedPtr tmpxz, double **h_tmpxz, MPI_File file) {
   long cnti, cntk;
   cudaMemcpy3DParms dencpy = { 0 };
   MPI_Offset fileoffset;
   double tmp[3];

   outdenxz_kernel<<<dimGrid3d, dimBlock3d>>>(psi, tmpxzy, outstpx, outstpz);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dy, tmpxzy, tmpxz, localNx / outstpx, Nz / outstpz, Ny);

   dencpy.srcPtr = tmpxz;
   dencpy.dstPtr = make_cudaPitchedPtr(h_tmpxz[0], Nz * sizeof(double), Nz, localNx);
   dencpy.extent = make_cudaExtent(Nz * sizeof(double) / outstpz, localNx / outstpx, 1);
   dencpy.kind = cudaMemcpyDeviceToHost;
   cudaCheckError(cudaMemcpy3D(&dencpy));

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         tmp[0] = x[offsetNx + cnti];
         tmp[1] = z[cntk];
         tmp[2] = h_tmpxz[cnti / outstpx][cntk / outstpz];
         MPI_File_write_at_all(file, fileoffset, &tmp, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
         fileoffset += 3 * sizeof(double);
      }
   }
}

__global__ void outdenxz_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpx, long outstpz) {
   long cnti, cntj, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Z * outstpx; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);

         for (cntk = TID_X * outstpz; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psi2row = get_double_tensor_row(psi2, cnti / outstpx, cntk / outstpz);

            tmp = psirow[cntk];
            psi2row[cntj] = tmp * tmp;
         }
      }
   }
}

void outdenyz(cudaPitchedPtr psi_t, cudaPitchedPtr tmpyzx, cudaPitchedPtr tmpyz, double **h_tmpyz, MPI_File file) {
   long cntj, cntk;
   cudaMemcpy3DParms dencpy = { 0 };
   MPI_Offset fileoffset;
   double tmp[3];

   transpose(tran_psi);

   outdenyz_kernel<<<dimGrid3d, dimBlock3d>>>(psi_t, tmpyzx, outstpy, outstpz);
   simpint3d_kernel<<<dimGrid2d, dimBlock2d>>>(dx, tmpyzx, tmpyz, localNy / outstpy, Nz / outstpz, Nx);

   dencpy.srcPtr = tmpyz;
   dencpy.dstPtr = make_cudaPitchedPtr(h_tmpyz[0], Nz * sizeof(double), Nz, localNy);
   dencpy.extent = make_cudaExtent(Nz * sizeof(double) / outstpz, localNy / outstpy, 1);
   dencpy.kind = cudaMemcpyDeviceToHost;
   cudaCheckError(cudaMemcpy3D(&dencpy));

   fileoffset = rank * 3 * sizeof(double) * (localNy / outstpy) * (Nz / outstpz);

   for (cntj = 0; cntj < localNy; cntj += outstpy) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         tmp[0] = y[offsetNy + cntj];
         tmp[1] = z[cntk];
         tmp[2] = h_tmpyz[cntj / outstpy][cntk / outstpz];
         MPI_File_write_at_all(file, fileoffset, &tmp, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
         fileoffset += 3 * sizeof(double);
      }
   }
}

__global__ void outdenyz_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpy, long outstpz) {
   long cnti, cntj, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Z; cnti < d_Nx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y * outstpy; cntj < d_localNy; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);

         for (cntk = TID_X * outstpz; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psi2row = get_double_tensor_row(psi2, cntj / outstpy, cntk / outstpz);

            tmp = psirow[cntk];
            psi2row[cnti] = tmp * tmp;
         }
      }
   }
}

void outpsi2xy(cudaPitchedPtr psi, cudaPitchedPtr tmpxy, double **h_tmpxy, MPI_File file) {
   long cnti, cntj;
   cudaMemcpy3DParms psi2cpy = { 0 };
   MPI_Offset fileoffset;
   double tmp[3];

   outpsi2xy_kernel<<<dimGrid2d, dimBlock2d>>>(psi, tmpxy, outstpx, outstpy);

   psi2cpy.srcPtr = tmpxy;
   psi2cpy.dstPtr = make_cudaPitchedPtr(h_tmpxy[0], Ny * sizeof(double), Ny, localNx);
   psi2cpy.extent = make_cudaExtent(Ny * sizeof(double) / outstpy, localNx / outstpx, 1);
   psi2cpy.kind = cudaMemcpyDeviceToHost;
   cudaCheckError(cudaMemcpy3D(&psi2cpy));

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Ny / outstpy);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         tmp[0] = x[offsetNx + cnti];
         tmp[1] = y[cntj];
         tmp[2] = h_tmpxy[cnti / outstpx][cntj / outstpy];
         MPI_File_write_at_all(file, fileoffset, &tmp, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
         fileoffset += 3 * sizeof(double);
      }
   }
}

__global__ void outpsi2xy_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpx, long outstpy) {
   long cnti, cntj;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Y * outstpx; cnti < d_localNx; cnti += GRID_STRIDE_Y) {
      for (cntj = TID_X * outstpy; cntj < d_Ny; cntj += GRID_STRIDE_X) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psi2row = get_double_matrix_row(psi2, cnti / outstpx);

         tmp = psirow[d_Nz2];
         psi2row[cntj / outstpy] = tmp * tmp;
      }
   }
}

void outpsi2xz(cudaPitchedPtr psi, cudaPitchedPtr tmpxz, double **h_tmpxz, MPI_File file) {
   long cnti, cntk;
   cudaMemcpy3DParms psi2cpy = { 0 };
   MPI_Offset fileoffset;
   double tmp[3];

   outpsi2xz_kernel<<<dimGrid2d, dimBlock2d>>>(psi, tmpxz, outstpx, outstpz);

   psi2cpy.srcPtr = tmpxz;
   psi2cpy.dstPtr = make_cudaPitchedPtr(h_tmpxz[0], Nz * sizeof(double), Nz, localNx);
   psi2cpy.extent = make_cudaExtent(Nz * sizeof(double) / outstpz, localNx / outstpx, 1);
   psi2cpy.kind = cudaMemcpyDeviceToHost;
   cudaCheckError(cudaMemcpy3D(&psi2cpy));

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         tmp[0] = x[offsetNx + cnti];
         tmp[1] = z[cntk];
         tmp[2] = h_tmpxz[cnti / outstpx][cntk / outstpz];
         MPI_File_write_at_all(file, fileoffset, &tmp, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
         fileoffset += 3 * sizeof(double);
      }
   }
}

__global__ void outpsi2xz_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpx, long outstpz) {
   long cnti, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Y * outstpx; cnti < d_localNx; cnti += GRID_STRIDE_Y) {
      psirow = get_double_tensor_row(psi, cnti, d_Ny2);
      psi2row = get_double_matrix_row(psi2, cnti / outstpx);

      for (cntk = TID_X * outstpz; cntk < d_Nz; cntk += GRID_STRIDE_X) {
         tmp = psirow[cntk];
         psi2row[cntk / outstpz] = tmp * tmp;
      }
   }
}

void outpsi2yz(cudaPitchedPtr psi, cudaPitchedPtr tmpyz, double **h_tmpyz, MPI_File file) {
   long cntj, cntk;
   int rankNx2;

   cudaMemcpy3DParms psi2cpy = { 0 };
   MPI_Offset fileoffset;
   double tmp[3];

   rankNx2 = Nx2 / localNx;


   if (rank == rankNx2) {
      outpsi2yz_kernel<<<dimGrid2d, dimBlock2d>>>(psi, tmpyz, outstpy, outstpz);

      psi2cpy.srcPtr = tmpyz;
      psi2cpy.dstPtr = make_cudaPitchedPtr(h_tmpyz[0], Nz * sizeof(double), Nz, Ny);
      psi2cpy.extent = make_cudaExtent(Nz * sizeof(double) / outstpz, Ny / outstpy, 1);
      psi2cpy.kind = cudaMemcpyDeviceToHost;
      cudaCheckError(cudaMemcpy3D(&psi2cpy));

      fileoffset = 0;

      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk += outstpz) {
            tmp[0] = y[cntj];
            tmp[1] = z[cntk];
            tmp[2] = h_tmpyz[cntj / outstpy][cntk / outstpz];
            MPI_File_write_at(file, fileoffset, &tmp, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
            fileoffset += 3 * sizeof(double);
         }
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void outpsi2yz_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpy, long outstpz) {
   long cntj, cntk;
   long offsetNx2;
   double *psirow, *psi2row;
   double tmp;

   offsetNx2 = d_Nx2 % d_localNx;

   for (cntj = TID_Y * outstpy; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
      psirow = get_double_tensor_row(psi, offsetNx2, cntj);
      psi2row = get_double_matrix_row(psi2, cntj / outstpy);

      for (cntk = TID_X * outstpz; cntk < d_Nz; cntk += GRID_STRIDE_X) {
         tmp = psirow[cntk];
         psi2row[cntk / outstpz] = tmp * tmp;
      }
   }
}

void outdenxyz(cudaPitchedPtr psi, cudaPitchedPtr tmpxyz, double ***h_tmpxyz, double *h_tmpz, MPI_File file) {
   long cnti, cntj, cntk;
   cudaMemcpy3DParms psi2copy = { 0 };
   MPI_Offset fileoffset;

   outdenxyz_kernel<<<dimGrid3d, dimBlock3d>>>(psi, tmpxyz, outstpx, outstpy, outstpz);

   psi2copy.srcPtr = tmpxyz;
   psi2copy.dstPtr = make_cudaPitchedPtr(h_tmpxyz[0][0], Nz * sizeof(double), Nz, Ny);
   psi2copy.extent = make_cudaExtent(Nz * sizeof(double) / outstpz, Ny / outstpy, localNx / outstpx);
   psi2copy.kind = cudaMemcpyDeviceToHost;
   cudaCheckError(cudaMemcpy3D(&psi2copy));

   fileoffset = rank * sizeof(double) * (localNx / outstpx) * (Ny / outstpy) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk += outstpz) {
            h_tmpz[cntk / outstpz] = h_tmpxyz[cnti / outstpx][cntj / outstpy][cntk / outstpz];
         }
         MPI_File_write_at_all(file, fileoffset, h_tmpz, Nz / outstpz, MPI_DOUBLE, MPI_STATUS_IGNORE);
         fileoffset += sizeof(double) * (Nz / outstpz);
      }
   }

   return;
}

__global__ void outdenxyz_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2, long outstpx, long outstpy, long outstpz) {
   long cnti, cntj, cntk;
   double *psirow, *psi2row;
   double tmp;

   for (cnti = TID_Z * outstpx; cnti < d_localNx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y * outstpy; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_double_tensor_row(psi, cnti, cntj);
         psi2row = get_double_tensor_row(psi2, cnti / outstpx, cntj / outstpy);

         for (cntk = TID_X * outstpz; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            tmp = psirow[cntk];
            psi2row[cntk / outstpz] = tmp * tmp;
         }
      }
   }
}
