#ifndef MATRIX_H 
#define MATRIX_H
#define DTYPE double
#define DSIZE sizeof(DTYPE)

struct Matrix {
  DTYPE* data;
  size_t w;
  size_t h;
};

#endif
