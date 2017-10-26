SRC = $(addprefix src/, matrix.c helpers.c)
OBJ = $(SRC:.c=.o)
CUSRC = $(SRC:.c=.cu)

all:
	cp src/matrix.c src/matrix.cu
	cp src/helpers.c src/helpers.cu
	nvcc $(CUSRC) -arch=sm_61 -O3 -o neo
	@rm $(CUSRC)

no-cuda:
	gcc src/matrix.c -DNO_CUDA -O3 -o neo_no_cuda

clean:
	@rm neo neo_no_cuda
