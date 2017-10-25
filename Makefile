all:
	cp src/matrix.c src/matrix.cu
	nvcc src/matrix.cu -arch=sm_61 -O3 -o neo
	rm src/matrix.cu

no-cuda:
	gcc src/matrix.c -DNO_CUDA -O3 -o neo_no_cuda

clean:
	@rm neo neo_no_cuda
