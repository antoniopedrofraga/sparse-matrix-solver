all: sequential openmp cuda

sequential:
	g++ -Wall -std=c++14 sequential.cpp */*.cpp -o ../bin/sequential
openmp:
	g++ -Wall -fopenmp -lpthread -std=c++14 openmp.cpp */*.cpp -o ../bin/openmp
cuda: cuda.o
	nvcc -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ cuda.cpp */*.cpp cuda.cu -o ../bin/cuda
run:
	rm -f ../outputs/*.csv
	../bin/sequential ../matrices/adder_dcop_32.mtx
	../bin/openmp ../matrices/adder_dcop_32.mtx
	../bin/cuda ../matrices/adder_dcop_32.mtx
runall:
	rm -f ../outputs/*.csv
	for m in $(wildcard ../matrices/*.mtx); \
		do printf "Running: ".$$m && printf "  Sequential " && ../bin/sequential $$m && printf " >>  OpenMP " && ../bin/openmp $$m && printf " >>  Cuda " && ../bin/cuda $$m && printf "  Done!\n"; \
	done \

