#ifndef CSR_H
#define CSR_H

#include "matrix.h"
#include <vector>

class CSR: public Matrix {
	int element_index; 
public:
	size_t irp_size;
	int nz, * ja, * irp;
	double * as;

	CSR(int cols, int rows, int nz);
	~CSR();
	void addPointer(int pointer);
	void addElement(int col_index, double value);
	
	int * getja();
	double * getas();
	int * getirp();
	void print();
};

#endif

