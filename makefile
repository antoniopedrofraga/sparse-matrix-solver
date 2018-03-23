all:
	nvcc -O3 -std=c++11 -Xcompiler -fopenmp -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ src/main.cpp src/*/*.cpp src/methods/cuda.cu -o bin/solver
run:
	rm -f outputs/*.csv
	bin/solver matrices/adder_dcop_32.mtx
runall:
	rm -f outputs/*.csv
	bin/solver $(wildcard matrices/*.mtx)

