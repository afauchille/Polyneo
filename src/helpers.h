#ifndef HELPHERS_H
# define HELPHERS_H
# include "matrix.h"

struct Matrix RandomMatrix(size_t w, size_t h);
struct Matrix UninitializedMatrix(size_t w, size_t h);
struct Matrix NoneMatrix();

int IsNoneMatrix(struct Matrix m);

/* Matrix conversion */
struct Matrix ToDevice(struct Matrix m);
struct Matrix ToHost(struct Matrix m);

int MatrixCmp(struct Matrix a, struct Matrix b);
void print_mat(float *matrix, size_t n);
void print_matrix(struct Matrix matrix);

#endif
