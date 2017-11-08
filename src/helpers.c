#include <stdlib.h>
#include <stdio.h>

#include "helpers.h"

struct Matrix GPUMatrix(size_t w, size_t h)
{
  struct Matrix m;
  m.w = w;
  m.h = h;
  cudaMalloc((void **)&(m.data), w * h * DSIZE);
  return m;
}

// This is a floating point data implementation. If DTYPE == int or similar the function has undefined behaviour
struct Matrix RandomMatrix(size_t w, size_t h)
{
  struct Matrix m = UninitializedMatrix(w, h);
  size_t n = w * h;
  for (size_t i = 0; i < n; ++i)
    m.data[i] = (DTYPE)rand() / (DTYPE)RAND_MAX;
  return m;
}

struct Matrix IdentityMatrix(size_t n)
{
  struct Matrix m;
  m.w = n;
  m.h = n;
  size_t nn = n * n;
  m.data = (DTYPE*)calloc(nn, DSIZE);
  for (int i = 0; i < nn; i += n + 1)
    m.data[i] = (DTYPE)1;
  return m;
}

struct Matrix ZeroMatrix(size_t w, size_t h)
{
  struct Matrix m;
  m.w = w;
  m.h = h;
  size_t n = w * h;
  m.data = (DTYPE*)calloc(n, DSIZE);
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

struct Matrix ToDevice(struct Matrix m)
{
  size_t size = m.w * m.h * DSIZE;
  struct Matrix res;
  res.w = m.w;
  res.h = m.h;
  cudaMalloc((void **) &res.data, size);
  cudaMemcpy(res.data, m.data, size, cudaMemcpyHostToDevice);
  return res;
}

struct Matrix ToHost(struct Matrix m)
{
  size_t size = m.w * m.h * DSIZE;
  struct Matrix res;
  res.w = m.w;
  res.h = m.h;
  res.data = (DTYPE*)malloc(size);
  cudaMemcpy(res.data, m.data, size, cudaMemcpyDeviceToHost);
  return res;
}

struct Matrix cp_gpu(struct Matrix a)
{
  struct Matrix res;
  const size_t size = a.w * a.h * DSIZE;
  cudaMalloc((void**)&(res.data), size);
  cudaMemcpy(res.data, a.data, size, cudaMemcpyHostToDevice);
  res.w = a.w;
  res.h = a.h;
  cudaCheckError();
  return res;
}

void GPUFree(struct Matrix m)
{
  cudaFree(m.data);
  cudaCheckError();
}

void CPUFree(struct Matrix m)
{
  free(m.data);
}

#define EPSILON double(0.0001)
// 1 => False, 0 => True
int MatrixCmp(struct Matrix a, struct Matrix b)
{
  if (a.w != b.w || a.h != b.h)
    return 1;
  size_t n = a.w * a.h;
  for (size_t i=  0; i < n; ++i)
    if (a.data[i] > b.data[i] + EPSILON || a.data[i] < b.data[i] - EPSILON)
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
  const size_t max_col = 10;
  size_t nb_lines = matrix.w < max_col ? matrix.w : max_col;
  for (size_t i = 0; i < matrix.w * matrix.h ; ++i)
  {
    if (i != 0 && i % nb_lines == 0)
      printf("\n%3f ", matrix.data[i]);
    else
      printf("%3f ", matrix.data[i]);
  }
  printf("\n\n");
}
