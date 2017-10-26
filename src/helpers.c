#include <stdlib.h>
#include <stdio.h>

#include "helpers.h"


float *RandomMatrix(size_t w, size_t h)
{
  size_t n = w * h;
  size_t size = n * sizeof(DTYPE);
  float *p = (float*)malloc(size);
  for (size_t i = 0; i < n; ++i)
    p[i] = (float)rand() / (float)RAND_MAX;
  return p;
}

float *UninitializedMatrix(size_t w, size_t h)
{
  size_t n = w * h;
  size_t size = n * sizeof(DTYPE);
  float *p = (float*)malloc(size);
  return p;
}

int MatrixCmp(const DTYPE *a, const DTYPE *b, size_t size)
{
  for (size_t i=  0; i < size; ++i)
    if (a[i] != b[i])
      return 1;
  return 0;
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
