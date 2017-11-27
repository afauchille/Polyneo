#ifndef CUPERF_H
#define CUPERF_H
#include "matrix.h"

struct Matrix mat_mult_cublas(struct Matrix a, struct Matrix b, double *time);

#endif
