#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "cuperf.h"

/* No Cuda device compatibility */
#ifdef NO_CUDA
  #define only_cuda(X)
#else
  #define only_cuda(X) X
#endif

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
  struct Matrix out = GPUMatrix(a.w, a.h);

  cudaCheckError();

  size_t n = a.w * a.h;
  int threads = 128;
  int blocks = (n + threads - 1) / threads;

  // Timer start
  CLOCK_START();

  add_k<<<blocks, threads>>>(a.data, b.data, out.data, n);
  cudaDeviceSynchronize();

  cudaCheckError();

  // Timer end
  CLOCK_STOP(time);

  return out;
}
#endif

/************************
* Scalar multiplication *
*************************/

/* CPU */

void _sc_mult_cpu_impl(struct Matrix a, struct Matrix out, DTYPE lambda, double *time)
{
  CLOCK_START();

  for (int i = 0; i < a.w * a.h; ++i)
    out.data[i] = a.data[i] * lambda;

  CLOCK_STOP(time);
}

struct Matrix sc_mult_cpu(struct Matrix a, DTYPE lambda, double *time)
{
  struct Matrix out = UninitializedMatrix(a.w, a.h);
  _sc_mult_cpu_impl(a, out, lambda, time);
  return out;
}

void sc_mult_cpu_in_place(struct Matrix a, DTYPE lambda, double *time)
{
  _sc_mult_cpu_impl(a, a, lambda, time);
}


/* GPU */

only_cuda(__global__
void sc_mult_k(const DTYPE *a, DTYPE lambda, DTYPE *out, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = lambda * a[i];
})

struct Matrix sc_mult_gpu(struct Matrix a, DTYPE lambda, double *time)
{
  struct Matrix out = GPUMatrix(a.w, a.h);
  const size_t n = a.w * a.h;

  int threads = 128;
  int blocks = (n + threads - 1) / threads;

  // Timer start
  CLOCK_START();

  sc_mult_k<<<blocks, threads>>>(a.data, lambda, out.data, n);
  cudaDeviceSynchronize();

  cudaCheckError();

  // Timer end
  CLOCK_STOP(time);

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

  CLOCK_START();

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

  CLOCK_STOP(time);

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
  assert(a.w == b.h);

  struct Matrix out = GPUMatrix(a.h, b.w);

  dim3 threads(16, 16);
  dim3 blocks((a.h + threads.x - 1) / threads.x, (b.w + threads.y - 1) / threads.y);

  // Timer start
  CLOCK_START();

  mat_mult_k<<<blocks, threads>>>(a, b, out);
  cudaDeviceSynchronize();

  cudaCheckError();

  // Timer end
  CLOCK_STOP(time);

  return out;
}

/**************************
* Determinant computation *
**************************/

__host__
struct Matrix lu(struct Matrix m)
{
  assert(m.w == m.h);
  // TODO: make ZeroMatrix for accurate return value
  struct Matrix l = IdentityMatrix(m.w);
  struct Matrix u = cp_cpu(m);

  // Trick for easily dividing first line by M[0,0]
  //sc_mult_cpu_in_place((struct Matrix){u.data, u.w, 1}, 1 / *u.data, NULL);

  for (size_t j = 1; j < u.h; ++j)
    {
      for (size_t i = 0; i < j; ++i)
        {
          DTYPE div = GET(u, i, i);
          assert(!FLOAT_EQ(div, 0));
          DTYPE pivot = GET(u, i, j) / div;
          SET(l, i, j, pivot);
          for (size_t k = 0; k < u.w; ++k)
            GET(u, k, j) -= pivot * GET(u, k, i);
        }
    }

  // Debug part (not useful because it works)
  //print_matrix(m);
  struct Matrix lu_res = mat_mult_cpu(l, u, NULL);
  //print_matrix(lu_res);

  if (MatrixCmp(m, lu_res))
    exit(2);/*
    printf("Success!\n");
  else
  printf("Faaaaaaaaaaaaiiiiiiil!\n");*/
  // not accurate: there is 1 more in each cell of the diagonal, because of l initialization
  struct Matrix ret = add_cpu(l, u, NULL);
  CPUFree(l);
  CPUFree(u);
  CPUFree(lu_res);
  return ret;
}

__host__
DTYPE det_cpu(struct Matrix m, double *time)
{
  assert(m.w == m.h);
  CLOCK_START();
  struct Matrix lu_mat = lu(m);
  CLOCK_STOP(time);
  DTYPE res = 0;
  for (size_t i = 0; i < m.w; ++i)
    res *= GET(m, i, i) - 1; // the -1 corrects the incorrect addition of l + u

  CPUFree(lu_mat);

  return res;
}

/*************
* Benchmarks *
*************/
void bench_mult()
{
  FILE *f = fopen("mult.bench", "w+");
  for (size_t n = 100; n <= 7000; n += 100)
    {
      double time;
      struct Matrix a = RandomMatrix(n, n);
      struct Matrix b = RandomMatrix(n, n);
      struct Matrix a_d = ToDevice(a);
      struct Matrix b_d = ToDevice(b);
      // Dont't need output
      GPUFree(mat_mult_gpu(a_d, b_d, &time));
      GPUFree(a_d);
      GPUFree(b_d);
      CPUFree(a);
      CPUFree(b);
      fprintf(f, "%d %f\n", n, time);
      fflush(f);
    }
  fclose(f);
}

void bench_det()
{
  for (size_t n = 1000; n <= 4000; n += 100)
    {
      double time;
      struct Matrix a = RandomMatrix(n, n);
      det_cpu(a, &time);
      printf("%d %f\n", n, time);
      CPUFree(a);
    }
}


#define PRINT 1
#define NO_PRINT 0

// print param isn't used but it could be
int check_mult(size_t N, int print)
{
  const char *comparaisons[3] = {"CPU", "cuBLAS", "cuPARSE"};

  struct Matrix a = RandomMatrix(N, N);
  struct Matrix b = RandomMatrix(N, N);

  /* Compare CPU & GPU */
  int result = compare_results(&mat_mult_cpu, &mat_mult_gpu, a, b, comparaisons[0]);
  CPUFree(a);
  CPUFree(b);
  return result;
}

int main(int argc, char **argv)
{
  /* No args */
  if (argc == 1)
    {
      const size_t N = 10;
      struct Matrix a = RandomMatrix(N, N);
      struct Matrix b = RandomMatrix(N, N);
      struct Matrix a_gpu = ToDevice(a);
      struct Matrix b_gpu = ToDevice(b);
      double t_cublas;
      double t_gpu = 0;
      struct Matrix c_cublas = ToHost(mat_mult_cublas(b_gpu, a_gpu, &t_cublas));
      struct Matrix c_gpu = mat_mult_gpu(a_gpu, b_gpu, &t_gpu);
      struct Matrix c = ToHost(c_gpu);
      int res = MatrixCmp(c, c_cublas);
      print_matrix(c_cublas);
      print_matrix(c);
      if (res)
        printf("Noob\n");
      printf("Cublas time: %fs\nOur time:    %fs\n", t_cublas, t_gpu);
      CPUFree(a);
      CPUFree(b);
      return res;
    }

  /* else if argc > 1 */
  int ret = 0;
  if (strcmp(argv[1], "check") == 0)
    ret = check_mult(5, NO_PRINT);
  else if (strcmp(argv[1], "bench") == 0)
    {
      if (argc < 2)
        {
          printf("Usage: ./neo bench [mult|det]");
          return 1;
        }
      if (strcmp(argv[2], "mult") == 0)
        bench_mult();
      else if (strcmp(argv[2], "det") == 0)
        bench_det();
    }
  return ret;
}
