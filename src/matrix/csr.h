#ifndef CSR_H
#define CSR_H

#include "matrix.h"
#include <vector>

class CSR: public Matrix {
	int element_index; 
	std::pair<double, double> cuda_times_csr;
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

	unsigned long long getScalarMegaFlops();
	unsigned long long getVecMinMegaFlops();
	void trackCSRTime(int method);
	void printElapsedCUDATime();
};

#endif

