#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "cuperf.h"

#define M 6
#define N 5
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

/* To use:
cublasDgeam pour add
cublasDgbmv pour mult
*/

/* Assuming a and b are gpu matrices */
struct Matrix mat_mult_cublas(struct Matrix a, struct Matrix b, double *time)
{
  assert(a.w == b.h);
  struct Matrix c = GPUMatrix(a.h, b.w);

  int lda = a.h, ldb = a.w, ldc = a.h;

  const double alpha = 1;
  const double beta = 0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  CLOCK_START();
  if (CUBLAS_STATUS_SUCCESS != cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.h, b.w, a.w, &alpha, a.data, lda, b.data, ldb, &beta, c.data, ldc))
    {
      printf("Error!\n");
      exit(2);
    }
  CLOCK_STOP(time);

  return c;
}
