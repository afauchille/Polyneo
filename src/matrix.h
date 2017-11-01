#ifndef MATRIX_H 
#define MATRIX_H

#include <stdio.h>

#include "helpers.h"
#include "tests.h"

#define DTYPE double
#define DSIZE sizeof(DTYPE)

/* Clock */

#define CLOCK_START() struct timespec t_start = {0, 0}, t_end = {0, 0};\
  clock_gettime(CLOCK_MONOTONIC, &t_start);

#define CLOCK_STOP(X) clock_gettime(CLOCK_MONOTONIC, &t_end);\
  *X = ((double)t_end.tv_sec + 1.0e-9 * t_end.tv_nsec) - ((double)t_start.tv_sec + 1.0e-9 * t_start.tv_nsec);

struct Matrix {
  DTYPE* data;
  size_t w;
  size_t h;
};

#endif
