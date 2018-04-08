#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <omp.h>
#include "../utils/utils.h"

class Matrix {
public:
	bool fits_in_memory;
	int cols, rows, nz;
	long long mflops;
	double start, done, mean, average_deviation;
	double elapsed_time, measures;
	std::pair<int, double> *omp_times_threads;
	bool measuring;
	double * x, * y;
	Matrix(int cols, int rows, int nz);
	void trackTimeOMP(int num_threads);
	double * getX();
	unsigned long long getMegaFlops();
	unsigned long long getMegaFlops(int i);
	int getnz();
	int getCols();
	int getRows();
	void printElapsedTime();
	void trackTime();
	void resetResults();
	void initVectors();

	bool fitsInMemory();
};

#endif
