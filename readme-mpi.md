## Instructions on compiling and installing MPI library for the use with DBEC-GP-MPI-CUDA programs.

These instructions should be followed only if MPI is not installed on the target computer cluster where you plan to run DBEC-GP-MPI-CUDA programs, or it was not compiled with the support for CUDA. We will give a short description on how to compile OpenMPI with support for CUDA; other MPI implementations may require different steps. List of CUDA-aware MPI implementations can be found [here](https://developer.nvidia.com/mpi-solutions-gpus).

If you are unsure whether OpenMPI installed on the target system already has support for CUDA, OpenMPI installation can be checked by following the instructions at [OpenMP CUDA FAQ](https://www.open-mpi.org/faq/?category=runcuda).

To compile and install CUDA-aware OpenMPI, follow these steps:

1. Download the latest [OpenMPI](https://www.open-mpi.org/). CUDA support in OpenMPI is available since version 1.8, however we recommend version 1.10 or higher.

2. Unpack the source tarball and go to the corresponding directory.

3. Examine the configuration parameters, and select the appropriate for the system on which DBEC-GP-MPI-CUDA programs will run. Most importantly, enable support for the interconnect used by the target system (e.g., InfiniBand). Detailed list of possible configure parameters can be found [here](https://www.open-mpi.org/faq/?category=building). If the OpenMPI was already installed, obtain the configure script that was
used to compile it and proceed to step 4.

4. Add the `--with-cuda` parameter to the configure script, for example:

        ./configure ... --with-cuda
   If CUDA-aware MPI is to be installed in a non-standard directory, its location may be specified after `--with-cuda` parameter, for example:

        ./configure ... --with-cuda=$HOME/openmpi

5. Build and install OpenMPI, by executing the following commands:

        make
        make install
   Note that you may need administrative privileges to run the second command, depending on the location of the installation.

After this, the CUDA-aware OpenMPI library is installed and may be used. If OpenMPI was installed alongside other MPI implementations, and is not used by default, you must change the makefile provided with the programs to use the newly created mpicc compiler with appropriate location of header files and libraries.
