#ifndef TESTS_H
#define TESTS_H

#include "matrix.h"

/* Color output */

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

int compare_results(
  struct Matrix (*fun_cpu)(struct Matrix, struct Matrix, double *),
  struct Matrix (*fun_gpu)(struct Matrix, struct Matrix, double *),
  struct Matrix a, struct Matrix b, const char *output_name);

#endif
