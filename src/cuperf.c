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

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p+1, &alpha, &m[IDX2F(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p+1, &beta, &m[IDX2F(p,q,ldm)], 1);
}

int init_matrix(cublasHandle_t *handleA, float** devPtrA, struct Matrix a)
{
  cudaError_t cudaStat;    
  cublasStatus_t stat;

  cudaStat = cudaMalloc ((void**)devPtrA, a.w*a.h*DSIZE/*sizeof(*(a.data))*/);
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  stat = cublasCreate(handleA);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (a.h, a.w, DSIZE /*sizeof(*(a.data))*/, a.data, a.h, *devPtrA, a.h);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (*devPtrA);
      cublasDestroy(*handleA);
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

struct Matrix add_cublas(struct Matrix a, struct Matrix b, double *time)
{
  /* cuBLAS Initialization */
  //cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;
  float* devPtrA;
  float* devPtrB;
  float* devPtrOut;

  /* cublas-geam Initialization (addition) */
  cublasOperation_t transa = CUBLAS_OP_N; // op(A) = A
  cublasOperation_t transb = CUBLAS_OP_N; // op(B) = B
  double alpha = 1; // scalar coefficient applied to A
  double beta = 1;  // scalar coefficient applied to B
  int ld = 1; // leading dimension of array

  int ret;
  int i, j;

  /* matrix Initialization */
  struct Matrix out = UninitializedMatrix(a.w, a.h);

  ret = init_matrix(&handle, &devPtrA, a);
  if (ret != EXIT_SUCCESS)
    printf("*** Problem initializing matrix");

  ret = init_matrix(&handle, &devPtrB, b);
  if (ret != EXIT_SUCCESS)
    printf("*** Problem initializing matrix");

  ret = init_matrix(&handle, &devPtrOut, out);
  if (ret != EXIT_SUCCESS)
    printf("*** Problem initializing matrix");

  /* Perform addition */
  CLOCK_START();

  stat = cublasDgeam(handle, transa, transb, a.h, a.w, &alpha, a.data, ld,
    &beta, b.data, ld, out.data, ld);

  CLOCK_STOP(time);

  /* Get data from cuBLAS */
  stat = cublasGetMatrix (out.h, out.w, DSIZE, devPtrOut, out.h, out.data, out.h);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("*** Data upload failed");
      cudaFree (devPtrOut);
      cublasDestroy(handle);        
  }    
  
  /* Un-initialize */
  cudaFree (devPtrA);
  cudaFree (devPtrB);
  cudaFree (devPtrOut);
  cublasDestroy(handle);

  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          printf ("%7.0f", out.data[IDX2F(i,j,M)]);
      }
      printf ("\n");
  }

  return out;
} 

int cu_test (void){
  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  float* devPtrA;
  float* a = 0;
  a = (float *)malloc (M * N * sizeof (*a));
  if (!a) {
      printf ("host memory allocation failed");
      return EXIT_FAILURE;
  }
  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          a[IDX2F(i,j,M)] = (float)((i-1) * M + j);
      }
  }
  cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (devPtrA);
      cublasDestroy(handle);
      return EXIT_FAILURE;
  }
  modify (handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
  stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data upload failed");
      cudaFree (devPtrA);
      cublasDestroy(handle);        
      return EXIT_FAILURE;
  }    
  cudaFree (devPtrA);
  cublasDestroy(handle);
  for (j = 1; j <= N; j++) {
      for (i = 1; i <= M; i++) {
          printf ("%7.0f", a[IDX2F(i,j,M)]);
      }
      printf ("\n");
  }
  free(a);
  return EXIT_SUCCESS;
}
