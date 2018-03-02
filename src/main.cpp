#include <iostream>
#include "matrix/csr.h"
#include "matrix/ellpack.h"
#include "solvers/solver.h"
#include "io/iomanager.h"
#include "matrix/matrix.h"

using namespace std;

int main(int argc, char ** argv) {	
	IOmanager * io = new IOmanager();
	for (int i = 1; i < argc; i++) {
		string path(argv[i]);
		pair<CSR*, Ellpack*> matrices = io->readFile(path);
		Solver * solver = new Solver(matrices);
		
		solver->sequential();
		solver->cuda();
		solver->openMP();
		
		cout << "Solved: " << path << endl;
	}
}
