#ifndef HELPHERS_H
#define HELPHERS_H
# include "matrix.h"

float *RandomMatrix(size_t w, size_t h);
float *UninitializedMatrix(size_t w, size_t h);
int MatrixCmp(const DTYPE *a, const DTYPE *b, size_t size);
void print_matrix(struct Matrix *matrix);
void print_mat(float *matrix, size_t n);

#endif
