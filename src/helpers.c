#include <stdlib.h>
#include <stdio.h>

#include "helpers.h"
#include "matrix.h"


float *RandomMatrix(size_t w, size_t h)
{
  size_t n = w * h;
  size_t size = n * sizeof(DTYPE);
  float *p = (float*)malloc(size);
  for (size_t i = 0; i < n; ++i)
    p[i] = (float)rand() / (float)RAND_MAX;
  return p;
}

void print_mat(float *matrix, size_t n)
{
  size_t nb_lines = 10;
  for (size_t i = 0; i < n; ++i)
  {
    if (i % nb_lines == 0)
      printf("\n%3f ", matrix[i]);
    else
      printf("%3f ", matrix[i]);
  }
  printf("\n");
}
