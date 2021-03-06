#ifndef HELPHERS_H
# define HELPHERS_H
# include "matrix.h"

#define EPSILON double(1e-5)
#define FLOAT_EQ(X, Y) (X < Y + EPSILON && X > Y - EPSILON)

struct Matrix GPUMatrix(size_t w, size_t h);

struct Matrix RandomMatrix(size_t w, size_t h);
struct Matrix IdentityMatrix(size_t n);
struct Matrix ZeroMatrix(size_t w, size_t h);
struct Matrix UninitializedMatrix(size_t w, size_t h);
struct Matrix NoneMatrix();

int IsNoneMatrix(struct Matrix m);

/* Matrix conversion */
struct Matrix ToDevice(struct Matrix m);
struct Matrix ToHost(struct Matrix m);
struct Matrix cp_gpu(struct Matrix a);
struct Matrix cp_cpu(struct Matrix a);
void GPUFree(struct Matrix m);
void CPUFree(struct Matrix m);

int MatrixCmp(struct Matrix a, struct Matrix b);
void print_mat(float *matrix, size_t n);
void print_matrix(struct Matrix matrix);

#endif
