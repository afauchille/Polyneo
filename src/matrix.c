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

#define cudaCheckError() {						\
    cudaError e = cudaGetLastError();					\
    if (e != cudaSuccess) {						\
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);						\
    }									\
  }


/***********
* Addition *
***********/

only_cuda(__host__)
struct Matrix add_cpu(struct Matrix a, struct Matrix b, double *time)
{
  assert(a.w == b.w && a.h == b.h);
  struct Matrix out = UninitializedMatrix(a.w, a.h);
  const size_t n = a.w * a.h;

  // Timer start
  CLOCK_START();
  for (size_t i = 0; i < n; i++)
    out.data[i] = a.data[i] + b.data[i];
  // Timer end
  CLOCK_STOP(time);
  return out;
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
struct Matrix add_gpu(struct Matrix a, struct Matrix b, double *time)
{
  assert(a.w == b.w && a.h == b.h);
  struct Matrix out = UninitializedMatrix(a.w, a.h);
  const size_t n = a.w * a.h;
  const size_t size = n * DSIZE;
  DTYPE *aG, *bG, *outG;
  cudaMalloc((void **) &aG, size);
  cudaMalloc((void **) &bG, size);
  cudaMalloc((void **) &outG, size);
  cudaMemcpy(aG, a.data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bG, b.data, size, cudaMemcpyHostToDevice);

  cudaCheckError();

  int threads = 128;
  int blocks = (n + threads - 1) / threads;

  // Timer start
  CLOCK_START();

  add_k<<<blocks, threads>>>(aG, bG, outG, n);
  cudaDeviceSynchronize();

  cudaCheckError();

  // Timer end
  CLOCK_STOP(time);

  cudaMemcpy(out.data, outG, size, cudaMemcpyDeviceToHost);
  cudaCheckError();
  return out;
}
#endif

/************************
* Scalar multiplication *
*************************/

// TODO: add in place version
only_cuda(__host__)
struct Matrix sc_mult_cpu(struct Matrix a, DTYPE lambda, double *time)
{
  struct Matrix out = UninitializedMatrix(a.w, a.h);
  for (int i = 0; i < a.w * a.h; ++i)
    out.data[i] = a.data[i] * lambda;
  return out;
}

only_cuda(__global__
void sc_mult_k(const DTYPE *a, DTYPE lambda, DTYPE *out, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = lambda * a[i];
})

struct Matrix sc_mult_gpu(struct Matrix a, DTYPE lambda, double *time)
{
  struct Matrix out = UninitializedMatrix(a.w, a.h);
  const size_t n = a.w * a.h;
  const size_t size = n * DSIZE;
  DTYPE *aG, *outG;
  cudaMalloc((void **) &aG, size);
  cudaMalloc((void **) &outG, size);
  cudaMemcpy(aG, a.data, size, cudaMemcpyHostToDevice);

  cudaCheckError();

  int threads = 128;
  int blocks = (n + threads - 1) / threads;

  // Timer start
  CLOCK_START();

  sc_mult_k<<<blocks, threads>>>(aG, lambda, outG, n);
  cudaDeviceSynchronize();

  cudaCheckError();

  // Timer end
  CLOCK_STOP(time);

  cudaMemcpy(out.data, outG, size, cudaMemcpyDeviceToHost);
  cudaCheckError();
  return out;
}

/*********************************
* Matrix & Vector multiplication *
**********************************/

// TODO: add in place version
only_cuda(__host__)
struct Matrix vec_mult_cpu(struct Matrix a, struct Matrix b, double *time)
{
  struct Matrix out = UninitializedMatrix(b.w, a.h);
  // Exception si a.h != b.w
  for (int i = 0; i < out.h; ++i)
    for (int j = 0; j < out.w; ++j)
    {
      float somme = 0;
      for (int k = 0; k < out.w; ++k)
      {
        int indice2 = out.h * k + i;
        int indice1 = out.h * j + k;
        printf("indice1: %d\tindice2: %d\n", indice1, indice2);
        somme += a.data[indice1] * b.data[indice2];
        printf("%f\n", somme);
      }
      out.data[out.h * j + i] = somme;
    }
  return out;
}

/**********
* Compare * 
***********/

int compare_results(
  struct Matrix (*fun_cpu)(struct Matrix, struct Matrix, double *),
  struct Matrix (*fun_gpu)(struct Matrix, struct Matrix, double *),
  struct Matrix a, struct Matrix b)
{
  /* Initialization */
  double time_cpu, time_gpu;
  
  a.data[0] = 1.0;
  a.data[1] = 2.0;
  a.data[2] = 3.0;
  b.data[0] = 1.0;
  b.data[1] = 0.0;
  b.data[4] = 0.0;
  b.data[7] = 1.0;
  b.data[3] = 1.0;
  b.data[6] = 1.0;

  printf("* Inputs:\n");
  print_matrix(a);
  print_matrix(b);

  /* Running */
  struct Matrix out_cpu = (*fun_cpu)(a, b, &time_cpu);
  struct Matrix out_gpu = (*fun_gpu)(a, b, &time_gpu);
  
  printf("* CPU output:\n");
  print_matrix(out_cpu);
  printf("* GPU output:\n");
  print_matrix(out_gpu);

  /* Display time & Output */
  printf("* Time taken:\nCPU: %fs\nGPU: %fs\n", time_cpu, time_gpu);

  int result = MatrixCmp(out_cpu, out_gpu);
  if (result == 0)
    printf("Output are the same!\n");
  else
    printf("*** Error: Outputs are differents.\n");
  return result;
}


int main(int argc, char **argv)
{
  struct Matrix a = RandomMatrix(N, N);
  struct Matrix b = RandomMatrix(N, N);

  /*
  struct Matrix out_cpu = UninitializedMatrix(N, N);
  struct Matrix out_gpu = UninitializedMatrix(N, N);

  add_cpu(a, b, out_cpu, &time_cpu);
  add_gpu(a, b, out_gpu, &time_gpu);
  vec_mult_cpu(a, b, out_cpu, N, &time_cpu);
  */

  return compare_results(&add_cpu, &add_gpu, a, b); 
}
