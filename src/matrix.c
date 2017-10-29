#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "matrix.h"
#include "helpers.h"

#define N 3

#ifdef NO_CUDA
  #define only_cuda(X)
#else
  #define only_cuda(X) X
#endif

#define CLOCK_START() struct timespec t_start = {0, 0}, t_end = {0, 0};\
  clock_gettime(CLOCK_MONOTONIC, &t_start);

#define CLOCK_STOP(X) clock_gettime(CLOCK_MONOTONIC, &t_end);\
  *X = ((double)t_end.tv_sec + 1.0e-9 * t_end.tv_nsec) - ((double)t_start.tv_sec + 1.0e-9 * t_start.tv_nsec);


/***********
* Addition *
***********/

only_cuda(__host__)
void add_cpu(struct Matrix a, struct Matrix b, struct Matrix out, double *time)
{
  assert(a.w == b.w && a.h == b.h);
  const size_t n = a.w * a.h;

  // Timer start
  CLOCK_START();
  for (size_t i = 0; i < n; i++)
    out.data[i] = a.data[i] + b.data[i];
  // Timer end
  CLOCK_STOP(time);
}

only_cuda(__global__
void add_k(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
})

#ifndef NO_CUDA
__host__
void add_gpu(struct Matrix a, struct Matrix b, struct Matrix out, double *time)
{
  assert(a.w == b.w && a.h == b.h);
  const size_t n = a.w * a.h;
  const size_t size = n * DSIZE;
  DTYPE *aG, *bG, *outG;
  cudaMalloc((void **) &aG, size);
  cudaMalloc((void **) &bG, size);
  cudaMalloc((void **) &outG, size);
  cudaMemcpy(aG, a.data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bG, b.data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(outG, out.data, size, cudaMemcpyHostToDevice);

  int threads = 128;
  int blocks = (n + threads - 1) / threads;

  // Timer start
  CLOCK_START();

  add_k<<<blocks, threads>>>(aG, bG, outG, n);
  // Timer end
  CLOCK_STOP(time);

  cudaMemcpy(out.data, outG, size, cudaMemcpyDeviceToHost);
}
#endif

/************************
* Scalar multiplication *
*************************/

// TODO: add in place version
only_cuda(__host__)
void sc_mult_cpu(struct Matrix a, DTYPE lambda, struct Matrix out, size_t n, double *time)
{
  for (int i = 0; i < n; ++i)
    out.data[i] = a.data[i] * lambda;
}


int main(int argc, char **argv)
{
  struct Matrix a = RandomMatrix(N, N);
  struct Matrix b = RandomMatrix(N, N);
  struct Matrix out_cpu = UninitializedMatrix(N, N);
  struct Matrix out_gpu = UninitializedMatrix(N, N);
  double time_cpu, time_gpu;
  print_matrix(a);
  print_matrix(b);
  add_cpu(a, b, out_cpu, &time_cpu);
  add_gpu(a, b, out_gpu, &time_gpu);
  print_matrix(out_cpu);
  print_matrix(out_gpu);

  printf("Time taken:\n- CPU: %fs\n- GPU: %fs\n", time_cpu, time_gpu);

  return MatrixCmp(out_cpu, out_gpu);
}
