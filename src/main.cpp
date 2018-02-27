#include <iostream>
#include <omp.h>
#include "solvers/solver.h"
#include "io/iomanager.h"
#include "matrix/matrix.h"

using namespace std;

int main(int argc, char ** argv) {	
	IOmanager * io = new IOmanager();
	for (int i = 1; i < argc; i++) {
		string path(argv[i]);
		Matrix * matrix = io->readFile(path);
		Solver * solver = new Solver(matrix);

		solver->cuda();
		solver->openMP();
	}
}
