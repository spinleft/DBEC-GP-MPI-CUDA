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

#include "cfg.h"

/**
 *    Configuration file parsing.
 *    cfg_file - stands for a configuration file, which is supplied on a command
 *    line
 */
int cfg_init(const char *cfg_file) {
   FILE *file;
   char buf[256];

   file = fopen(cfg_file, "r");
   if (! file) return 0;

   cfg_size = 0;
   while (fgets(buf, 256, file) != NULL) {
      if (sscanf(buf, "%s = %s", cfg_key[cfg_size], cfg_val[cfg_size]) == 2)
         cfg_size ++;
   }

   fclose(file);

   return cfg_size;
}

/**
 *    Configuration property value.
 *    key - property
 */
char *cfg_read(const char *key) {
   int i;

   for (i = 0; i < cfg_size; i ++)
      if (! strcmp(key, cfg_key[i])) return cfg_val[i];

   return NULL;
}
