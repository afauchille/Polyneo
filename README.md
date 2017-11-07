# Polyneo

Matrix operations library implementation on GPU using Cuda-C

Usage: make && ./neo

!!! This is a demo version, the library is not yet fully fonctionnal !!!

Operations supported :
[X] Scalar product : CPU & GPU
[X] Matrix addition : CPU & GPU
[X] Matrix multiplication : CPU & GPU
[ ] Determinant computaion
[ ] Matrix inversion
[ ] Singular value decomposition
[ ] Back substitution

The project will include the same operations on CPU to compare performance and resource utilization (memory, computing time, ...)
[X] Timer

When relevant, different algorithms could be implemented for the same operation.

We aim to also compare our performances against cuBLAS implementation.
