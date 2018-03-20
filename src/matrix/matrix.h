#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <chrono>

class Matrix {
	int cols, rows, flops, nz;
	std::chrono::high_resolution_clock::time_point start, done;
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
