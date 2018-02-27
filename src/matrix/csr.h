#ifndef CSR_H
#define CSR_H

#include "matrix.h"
#include <vector>

class CSR: public Matrix {
	std::vector<int> irp;
	int nz, * ja;
	double * as;
public:
	CSR(int cols, int rows, int nz);		
};

#endif

