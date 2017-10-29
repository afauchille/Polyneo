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

only_cuda(__host__)
void add_cpu(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n, double *time)
{
  // Timer start
  CLOCK_START()
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] + b[i];
  // Timer end
  CLOCK_STOP(time)
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
void add_gpu(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n, double *time)
{
  const int size = n * sizeof(DTYPE);
  DTYPE *aG, *bG, *outG;
  cudaMalloc((void **) &aG, size);
  cudaMalloc((void **) &bG, size);
  cudaMalloc((void **) &outG, size);
  cudaMemcpy(aG, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bG, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(outG, out, size, cudaMemcpyHostToDevice);

  int threads = 128;
  int blocks = (n + threads - 1) / threads;
  // Timer start
  CLOCK_START()
  add_k<<<blocks, threads>>>(aG, bG, outG, n);
  // Timer end
  CLOCK_STOP(time)

  cudaMemcpy(out, outG, size, cudaMemcpyDeviceToHost);
}
#endif

int main(int argc, char **argv)
{
  float *a = RandomMatrix(N, N);
  float *b = RandomMatrix(N, N);
  float *out_cpu = UninitializedMatrix(N, N);
  float *out_gpu = UninitializedMatrix(N, N);
  double time_cpu, time_gpu;
  print_mat(a, N * N);
  print_mat(b, N * N);
  add_cpu(a, b, out_cpu, N * N, &time_cpu);
  add_gpu(a, b, out_gpu, N * N, &time_gpu);
  print_mat(out_cpu, N * N);
  print_mat(out_gpu, N * N);

  printf("Time taken:\n- CPU: %fs\n- GPU: %fs\n", time_cpu, time_gpu);

  return MatrixCmp(out_cpu, out_gpu, N * N);
}
