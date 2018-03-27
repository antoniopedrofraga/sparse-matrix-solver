#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <omp.h>
#include "../utils/utils.h"

class Matrix {
	int cols, rows, nz;
	long long mflops;
	double start, done;
	double elapsed_time, measures;
	std::pair<int, double> *omp_times_threads;
	bool measuring;
public:
	double * x, * y;
	Matrix(int cols, int rows, int nz);
	void trackTimeOMP(int num_threads);
	double * getX();
	unsigned long long getMegaFlops();
	unsigned long long getMegaFlops(int i);
	int getnz();
	int getCols();
	void printElapsedTime();
	void trackTime();
	void resetResults();
};

#endif
