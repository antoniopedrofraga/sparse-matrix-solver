all: *.cpp */*.cpp
	g++ -Wall -fopenmp -lpthread -std=c++14 *.cpp */*.cpp -o ../bin/solver
run:
	../bin/solver ../matrices/adder_dcop_32.mtx
