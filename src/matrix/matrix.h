#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>

class Matrix {
	int cols, rows;
public:
	double * x, * y;
	Matrix(int cols, int rows);
	double * getX();
	int getCols();
};

#endif
