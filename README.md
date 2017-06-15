# DBEC-GP-MPI-CUDA

**DBEC-GP-MPI-CUDA** is a set of CUDA/MPI programs that solves the time-(in)dependent Gross-Pitaevskii nonlinear partial differential equation for BECs with contact and dipolar interaction in three spatial dimensions in a trap using imaginary-time and real-time propagation, using Nvidia GPUs. The Gross-Pitaevskii equation describes the properties of a dilute trapped Bose-Einstein condensate. The equation is solved using the split-step Crank-Nicolson method by discretizing space and time, as described in Ref. [1]. The discretized equation is then propagated in imaginary or real time over small time steps. Additional details, for the case of pure contact interaction, are given in Refs. [R2-R5].

[R1] [R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.](https://doi.org/10.1016/j.cpc.2015.03.024)  
[R2] [P. Muruganandam and S. K. Adhikari, Comput. Phys. Commun. 180 (2009) 1888.](https://doi.org/10.1016/j.cpc.2009.04.015)  
[R3] [D. Vudragovic et al., Comput. Phys. Commun. 183 (2012) 2021.](https://doi.org/10.1016/j.cpc.2012.03.022)  
[R4] [B. Sataric et al., Comput. Phys. Commun. 200 (2016) 411.](https://doi.org/10.1016/j.cpc.2015.12.006)  
[R5] [L. E. Young-S. et al., Comput. Phys. Commun. 204 (2016) 209.](https://doi.org/10.1016/j.cpc.2016.03.015)

## Description of DBEC-GP-MPI-CUDA code distribution

### I) Source codes

Programs are written in CUDA C programming language and are located in the [src](src/) folder, which has the following structure:

 - [src/imag3d-mpicuda](src/imag3d-mpicuda/) program solves the imaginary-time dipolar Gross-Pitaevskii equation in three spatial dimensions in an anisotropic harmonic trap.
 - [src/real3d-mpicuda](src/real3d-mpicuda/) program solves the real-time dipolar Gross-Pitaevskii equation in three spatial dimensions in an anisotropic harmonic trap.
 - [src/utils](src/utils/) provides utility functions for transposing data, parsing of configuration files, integration and differentiation, as well as allocation/deallocation of memory.

### II) Input parameters

For each program, a specific parameter input file has to be given on the command line. Examples of input files with characteristic set of options and their detailed descriptions are provided in the [input](input/) folder.

Additionally, the folder [test](test/) contains input files used for testing of the programs, whose results are presented in the paper [1].

### III) Examples of matching outputs

The [output](output/) folder contains examples of matching outputs for all programs and default inputs available in the DBEC-GP-MPI-CUDA distribution. Some large density files are omitted to save space.

### IV) Compilation

Programs are compiled via a provided `makefile`.

The use of the makefile:

    make <target>

where possible targets are:

    all, clean

as well as program-specific targets, which compile only a specified program:

    imag3d-mpicuda, real3d-mpicuda

The provided makefile allows compilation of the DBEC-GP-MPI-CUDA programs, which rely on `nvcc` compiler being installed on the system. Nvidia's `nvcc` compiler is provided with CUDA and does not have to be installed separately. Additionally, a CUDA-aware MPI implementation must be installed. Before attempting a compilation, check and adjust (if necessary) variables CUDA_`CCFLAGS`, `CUDA_LIBS`, `MPI_PATH`, and `MPI_LIBS` in the `makefile`. We have tested the programs with OpenMPI 1.10. Instructions on how to compile CUDA-aware OpenMPI are given in the file [readme-mpi.md](readme-mpi.md).

**Examples of compiling:**

1. Compile all DBEC-GP-MPI-CUDA programs:

        make all

2. Compile only `imag3d-mpicuda` program:

        make imag3d-mpicuda

### V) Running the compiled programs

To run any of the CUDA/MPI programs compiled with the make command, you need to use the
syntax:

    mpiexec -np <nprocs> ./<programname> -i <parameterfile>

where `<nprocs>` is the number of processes invoked, `<programname>` is a name of the compiled executable, and `<parameterfile>` is a parameter input file prepared by the user. Examples of parameter input files are described in section II above, and are provided in the folder [input](input/). Matching output of the principal output files are given in the folder [output](output/); however, large density output files are omitted.

**Example of running a program:**

Run `imag3d-mpicuda` compiled program with the parameter input file `input/imag3d-input`, in
4 parallel processes:

    mpiexec -np 4 ./imag3d-mpicuda -i input/imag3d-input

**Important:**

You have to ensure that on each cluster node there are as many MPI processes as GPU cards available. Usually, this is managed through the submission file to the batch system on the cluster. For example, if OpenMPI is used, this can be achieved by the switches `--map-by ppr:n:node --bind-to none`, where `n` is the number of GPU cards per cluster node. For instance, if there is one GPU card per node, the above example would be executed as:

    mpiexec -np 4 --map-by ppr:1:node --bind-to none ./imag3d-mpicuda -i input/imag3d-input

For other implementations of MPI you should consult their manual pages, as well as user
guides for your cluster.

### VI) Authors

**DBEC-GP-MPI-CUDA** programs are developed by:

Vladimir Lončar, Antun Balaž *(Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)*  
Srđan Skrbić *(Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia)*  
Paulsamy Muruganandam *(Bharathidasan University, Tamil Nadu, India)*  
Luis E. Young-S, Sadhan K. Adhikari *(UNESP - Sao Paulo State University, Brazil)*  

Public use and modification of these codes are allowed provided that the following papers are cited:  
[1] [V. Loncar et al., Comput. Phys. Commun. 209 (2016) 190.](https://doi.org/10.1016/j.cpc.2016.07.029)  
[2] [V. Loncar et al., Comput. Phys. Commun. 200 (2016) 406.](https://doi.org/10.1016/j.cpc.2015.11.014)  
[3] [R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.](https://doi.org/10.1016/j.cpc.2015.03.024)

The authors would be grateful for all information and/or comments regarding the use of the programs.

### VII) Licence

**DBEC-GP-MPI-CUDA** code distribution is licensed under Apache License, Version 2.0. See [LICENCE](LICENCE) for details.
