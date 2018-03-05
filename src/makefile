all: *.cpp */*.cpp
	g++ -Wall -std=c++14 sequential.cpp */*.cpp -o ../bin/sequential
	g++ -Wall -fopenmp -lpthread -std=c++14 openmp.cpp */*.cpp -o ../bin/openmp
run:
	../bin/sequential ../matrices/adder_dcop_32.mtx
	../bin/openmp ../matrices/adder_dcop_32.mtx
runall:
	for m in $(wildcard ../matrices/*.mtx); \
		do printf "Running: ".$$m && ../bin/sequential $$m && ../bin/openmp $$m && printf " Done!\n"; \
	done \

