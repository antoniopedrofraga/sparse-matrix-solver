#ifndef SOLVER_H
#define SOLVER_H

#include "../matrix/matrix.h"

class Solver {
	Matrix * matrix;
public:
	Solver(Matrix * matrix);
	void cuda();
	void openMP();

};

#endif
