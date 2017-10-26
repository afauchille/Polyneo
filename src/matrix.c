#include <stdlib.h>

#include "matrix.h"
#include "helpers.h"

#define N 42
#ifdef NO_CUDA
  #define only_cuda(X)
#else
  #define only_cuda(X) X
#endif


only_cuda(__host__)
void add_cpu(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n)
{
  for (size_t i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}

only_cuda(__global__
void add_k(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n)
{
  size_t i = threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
})

#ifndef NO_CUDA
__host__
void add_gpu(const DTYPE *a, const DTYPE *b, DTYPE *out, size_t n)
{
  const int size = n * sizeof(DTYPE);
  DTYPE *aG, *bG, *outG;
  cudaMalloc((void **) &aG, size);
  cudaMalloc((void **) &bG, size);
  cudaMalloc((void **) &outG, size);
  cudaMemcpy(aG, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bG, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(outG, out, size, cudaMemcpyHostToDevice);

  // Timer start
  add_k<<<n, 1>>>(aG, bG, outG, n);
  // Timer end

  cudaMemcpy(out, outG, size, cudaMemcpyDeviceToHost);
}
#endif

int main(int argc, char **argv)
{
  float *a = RandomMatrix(N, N);
  float *b = RandomMatrix(N, N);
  a[0] = 1;
  b[0] = 2;
  print_mat(a, N);
  print_mat(b, N);
  float *out = RandomMatrix(N, N);
  add_cpu(a, b, out, N);
  print_mat(out, N);

  return 0;
}
