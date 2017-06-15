# DBEC-GP-OMP-CUDA-MPI programs are developed by:
#
# Vladimir Loncar, Antun Balaz
# (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
#
# Srdjan Skrbic
# (Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia)
#
# Paulsamy Muruganandam
# (Bharathidasan University, Tamil Nadu, India)
#
# Luis E. Young-S, Sadhan K. Adhikari
# (UNESP - Sao Paulo State University, Brazil)
#
#
# Public use and modification of these codes are allowed provided that the
# following papers are cited:
# [1] V. Loncar et al., Comput. Phys. Commun. 209 (2016) 190.      
# [2] V. Loncar et al., Comput. Phys. Commun. 200 (2016) 406.      
# [3] R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.
#
# The authors would be grateful for all information and/or comments
# regarding the use of the programs.

CUDA_CC = nvcc
# You can optionally set higher arch flag if your GPU supports it
CUDA_CCFLAGS = -arch=sm_20
CUDA_LIBS = -lcudart -lcufft -lmpi

# Set the following two variables to point to your custom CUDA-aware
# MPI implementation if you use it, otherwise leave them empty.
# For example:
# MPI_PATH=-I$(HOME)/openmpi/include
# MPI_LIBS=-L$(HOME)/openmpi/lib
MPI_PATH=
MPI_LIBS=

all: imag3d-mpicuda real3d-mpicuda
	rm -rf *.o

imag3d-mpicuda: diffint cfg mem tran
	$(CUDA_CC) $(CUDA_CCFLAGS) $(MPI_PATH) -dc src/imag3d-mpicuda/imag3d-mpicuda.cu -o imag3d-mpicuda.o
	$(CUDA_CC) $(CUDA_CCFLAGS) $(MPI_PATH) -o imag3d-mpicuda diffint.o cfg.o mem.o tran.o imag3d-mpicuda.o $(CUDA_LIBS) $(MPI_LIBS)

real3d-mpicuda: diffint cfg mem tran
	$(CUDA_CC) $(CUDA_CCFLAGS) $(MPI_PATH) -dc src/real3d-mpicuda/real3d-mpicuda.cu -o real3d-mpicuda.o
	$(CUDA_CC) $(CUDA_CCFLAGS) $(MPI_PATH) -o real3d-mpicuda diffint.o cfg.o mem.o tran.o real3d-mpicuda.o $(CUDA_LIBS) $(MPI_LIBS)

diffint:
	$(CUDA_CC) $(CUDA_CCFLAGS) -c src/utils/diffint.cu -o diffint.o

cfg:
	$(CUDA_CC) $(CUDA_CCFLAGS) -c src/utils/cfg.c -o cfg.o

mem:
	$(CUDA_CC) $(CUDA_CCFLAGS) -dc src/utils/mem.c -o mem.o

tran:
	$(CUDA_CC) $(CUDA_CCFLAGS) $(MPI_PATH) -c src/utils/tran.c -o tran.o

clean:
	rm -rf *.o imag3d-mpicuda real3d-mpicuda
