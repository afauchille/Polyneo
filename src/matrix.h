#ifndef MATRIX_H 
#define MATRIX_H
#define DTYPE float
#define DSIZE sizeof(DTYPE)

struct Matrix {
  float* data;
  size_t w;
  size_t h;
} ;

#endif
