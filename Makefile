SRC = $(addprefix src/, matrix.c helpers.c cuperf.c tests.c)
OBJ = $(SRC:.c=.o)
CUSRC = $(SRC:.c=.cu)
CULIBS = -lcublas

all:
	@cp src/matrix.c src/matrix.cu
	@cp src/helpers.c src/helpers.cu
	@cp src/cuperf.c src/cuperf.cu
	@cp src/tests.c src/tests.cu
	nvcc $(CULIBS) $(CUSRC) -arch=sm_61 -O3 -o neo -g
	@rm $(CUSRC)

check: all
	./neo check

no-cuda:
	gcc src/matrix.c -DNO_CUDA -O3 -o neo_no_cuda

memcheck: all
	valgrind --leak-check=full ./neo

clean:
	@rm neo neo_no_cuda
