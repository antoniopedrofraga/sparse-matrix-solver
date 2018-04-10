.PHONY: tests

all:
	nvcc -O3 -std=c++11 -arch=sm_35 -rdc=true -Xcompiler -fopenmp -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -I/apps/software/CUDA/8.0.44/samples/common/inc/ src/main.cpp src/*/*.cpp src/methods/cuda.cu -o bin/solver
memcheck:
	valgrind --tool=memcheck --leak-check=full -v bin/solver matrices/adder_dcop_32.mtx matrices/amazon0302.mtx
tests:
	nvcc -std=c++11 -Xcompiler -fopenmp,-g -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -I/apps/software/CUDA/8.0.44/samples/common/inc/ tests/*.cpp src/*/*.cpp src/methods/cuda.cu -o bin/tests
run-tests:
	./bin/tests
run:
	rm -f outputs/*.csv
	bin/solver matrices/cage4.mtx
runall:
	rm -f outputs/*.csv
	bin/solver $(wildcard matrices/*.mtx)

