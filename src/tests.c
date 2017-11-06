#include "tests.h"

/**********************
* Compare 2 functions *
***********************/

int compare_results(
  struct Matrix (*fun_cpu)(struct Matrix, struct Matrix, double *),
  struct Matrix (*fun_gpu)(struct Matrix, struct Matrix, double *),
  struct Matrix a, struct Matrix b, const char *output_name)
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
  struct Matrix a_d = ToDevice(a);
  struct Matrix b_d = ToDevice(b);
  struct Matrix out_d = (*fun_gpu)(a_d, b_d, &time_gpu);
  struct Matrix out_gpu = ToHost(out_d);
  GPUFree(a_d);
  GPUFree(b_d);
  GPUFree(out_d);

  /*
  printf("* %s output:\n", output_name);
  print_matrix(out_cpu);
  printf("* GPU output:\n");
  print_matrix(out_gpu);
  */

  /* Display time & Output */
  printf("* Time taken:\n%s: %fs\nGPU: %fs\n", output_name, time_cpu, time_gpu);

  int result = MatrixCmp(out_cpu, out_gpu);
  CPUFree(out_cpu);
  CPUFree(out_gpu);

  if (result == 0)
  {
    printf(ANSI_COLOR_GREEN);
    printf("[OK] Output are the same!");
  }
  else
  {
    printf(ANSI_COLOR_RED);
    printf("[KO] Outputs are differents.");
  }

  if (time_gpu < time_cpu)
  {
    printf(ANSI_COLOR_GREEN);
    printf("\t[OK] GPU is fastest than %s!\n", output_name);
  }
  else
  {
    printf(ANSI_COLOR_RED);
    printf("\t[KO] %s is fastest than GPU\n", output_name);
  }

  printf(ANSI_COLOR_RESET);
  return result;
}
