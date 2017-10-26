#include <stdlib.h>

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
