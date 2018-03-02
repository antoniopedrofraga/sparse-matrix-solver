#include "solver.h"
#include <omp.h>

Solver::Solver(std::pair<CSR*, Ellpack*> &matrices) {
	this->csr = matrices.first;
	this->ellpack = matrices.second;
}

void Solver::sequential() {
	
}

void Solver::cuda() {
	
}

void Solver::openMP() {
	
}
