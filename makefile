all: sequential openmp cuda

sequential:
	g++ -O3 -std=c++11 -Wall -fopenmp src/sequential.cpp src/*/*.cpp -o bin/sequential
openmp:
	g++ -O3 -std=c++11 -Wall -fopenmp -lpthread src/openmp.cpp src/*/*.cpp -o bin/openmp
cuda:
	nvcc -O3 -std=c++11 -Xcompiler -fopenmp -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ src/cuda.cpp src/*/*.cpp src/cuda.cu -o bin/cuda
run:
	rm -f outputs/*.csv
	bin/sequential matrices/adder_dcop_32.mtx
	bin/openmp matrices/adder_dcop_32.mtx
	bin/cuda matrices/adder_dcop_32.mtx
runall:
	rm -f outputs/*.csv
	for m in $(wildcard matrices/*.mtx); \
		do printf "Running: ".$$m && printf "  Sequential " && bin/sequential $$m && printf " >>  OpenMP " && bin/openmp $$m && printf " >>  Cuda " && bin/cuda $$m && printf "  Done!\n"; \
	done \

