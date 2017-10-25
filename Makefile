CC=nvcc

all:
	cp src/matrix.c src/matrix.cu
	nvcc src/matrix.cu -arch=sm_61 -O3 -o neo
	rm src/matrix.cu
