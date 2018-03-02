#include "solver.h"

Solver::Solver(std::pair<CSR*, Ellpack*> &matrices) {
	this->csr = matrices.first;
	this->ellpack = matrices.second;
}

void Solver::cuda() {

}

void Solver::openMP() {

}
