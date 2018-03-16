#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>

class Matrix {
	int cols, rows, flops, nz;
	clock_t start;
	double elapsed_time, measures;
	bool measuring;
public:
	double * x, * y;
	Matrix(int cols, int rows, int nz);
	double * getX();
	int getFlops();
	int getnz();
	int getCols();
	void trackTime();
};

#endif
