#include <stdlib.h>
#include <stdio.h>

#include "helpers.h"

// This is a floating point data implementation. If DTYPE == int or similar the function has undefined behaviour
struct Matrix RandomMatrix(size_t w, size_t h)
{
  struct Matrix m;
  m.w = w;
  m.h = h;
  size_t n = w * h;
  m.data = (DTYPE*)malloc(n * DSIZE);
  for (size_t i = 0; i < n; ++i)
    m.data[i] = (DTYPE)rand() / (DTYPE)RAND_MAX;
  return m;
}

struct Matrix UninitializedMatrix(size_t w, size_t h)
{
  struct Matrix m;
  m.w = w;
  m.h = h;
  size_t n = w * h;
  m.data = (DTYPE*)malloc(n * DSIZE);
  return m;
}

// A NULL-equivalent Matrix. Some functions accept a NoneMatrix as one of its argument, that modifies its behaviour.
struct Matrix NoneMatrix()
{
  struct Matrix m;
  m.data = NULL;
  m.w = 0;
  m.h = 0;
  return m;
}

int IsNoneMatrix(struct Matrix m)
{
  return m.data == NULL && m.w == 0 && m.h == 0;
}


// 1 => False, 0 => True
int MatrixCmp(struct Matrix a, struct Matrix b)
{
  if (a.w != b.w || a.h != b.h)
    return 1;
  size_t n = a.w * a.h;
  for (size_t i=  0; i < n; ++i)
    if (a.data[i] != b.data[i])
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

void print_matrix(struct Matrix matrix)
{
  //size_t nb_lines = matrix->w;
  size_t nb_lines = 10;
  for (size_t i = 0; i < matrix.w * matrix.h ; ++i)
  {
    if (i % nb_lines == 0)
      printf("\n%3f ", matrix.data[i]);
    else
      printf("%3f ", matrix.data[i]);
  }
  printf("\n");
}
