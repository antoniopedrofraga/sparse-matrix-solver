#ifndef CSR_H
#define CSR_H

#include "matrix.h"
#include <vector>

class CSR: public Matrix {
	std::vector<int> irp;
	int nz, * ja;
	double * as;
	
	int element_index; 
public:
	CSR(int cols, int rows, int nz);
	void addPointer(int pointer);
	void addElement(int col_index, double value);
	
	int * getja();
	double * getas();
	std::vector<int> getirp();
};

#endif

