#ifndef SOLVER_H
#define SOLVER_H

#include "../matrix/csr.h"
#include "../matrix/ellpack.h"

class Solver {
	CSR * csr;
	Ellpack * ellpack;
public:
	Solver(std::pair<CSR*, Ellpack*> &matrices);
	void sequential();
	void cuda();
	void openMP();

};

#endif
