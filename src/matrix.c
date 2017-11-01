#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "cuperf.h"

#define N 3

#ifdef NO_CUDA
  #define only_cuda(X)
#else
  #define only_cuda(X) X
#endif

#define cudaCheckError() {						\
    cudaError e = cudaGetLastError();					\
    if (e != cudaSuccess) {						\
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);						\
    }									\
  }

/* Matrix indexing */

#define GET(M, X, Y) M.data[M.w * Y + X]
#define SET(M, X, Y, DATA) M.data[M.w * Y + X] = DATA

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

/************************
* Vector multiplication *
*************************/

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

/************************
* Matrix multiplication *
*************************/

only_cuda(__host__)
struct Matrix mat_mult_cpu(struct Matrix a, struct Matrix b, double *time)
{
  assert(a.w == b.h);
  struct Matrix out = UninitializedMatrix(a.h, b.w);
  for (size_t i = 0; i < a.h; ++i)
    {
      for (int j = 0; j < b.w; ++j)
        {
          DTYPE res = 0;
          for (int k = 0; k < a.w; ++k)
            res += GET(a, k, j) * GET(b, i, k);
          SET(out, i, j, res);
        }
    }
  return out;
}

only_cuda(__global__
void mat_mult_k(struct Matrix a, struct Matrix b, struct Matrix out)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= out.w || j >= out.h)
    return;

  DTYPE res = 0;
  for (size_t k = 0; k < a.w; k++)
    res += GET(a, k, j) * GET(b, i, k);
  SET(out, i, j, res);
})

only_cuda(__host__)
struct Matrix mat_mult_gpu(struct Matrix a, struct Matrix b, double *time)
{
  const size_t n = a.w * a.h;
  const size_t size = n * DSIZE;
  DTYPE *aG, *bG, *outG;
  cudaMalloc((void **) &aG, size);
  cudaMalloc((void **) &bG, size);
  cudaMalloc((void **) &outG, size);
  cudaMemcpy(aG, a.data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bG, b.data, size, cudaMemcpyHostToDevice);
  struct Matrix out = UninitializedMatrix(a.h, b.w);

  cudaCheckError();

  // TODO: more dim
  //int threads = 128;
  //int blocks = (n + threads - 1) / threads;

  // Timer start
  CLOCK_START();

  //mat_mult_k<<<blocks, threads>>>(aG, bG, outG);
  cudaDeviceSynchronize();

  cudaCheckError();

  // Timer end
  CLOCK_STOP(time);

  cudaMemcpy(out.data, outG, size, cudaMemcpyDeviceToHost);
  cudaCheckError();

  return out;
}

int main(int argc, char **argv)
{
  struct Matrix a = RandomMatrix(N, N);
  struct Matrix b = RandomMatrix(N, N);
  const char *comparaisons[3] = {"CPU", "cuBLAS", "cuPARSE"};

  /* Compare CPU & GPU */
  compare_results(&add_cpu, &add_gpu, a, b, comparaisons[0]); 
  printf("\n");
  /* Compare cuBLAS & GPU */
  return compare_results(&add_cublas, &add_gpu, a, b, comparaisons[1]); 
}
