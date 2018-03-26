#include "matrix.h"
#include <time.h>
#include <iostream>
#include <chrono>

Matrix::Matrix(int cols, int rows, int nz) {
	this->nz = nz;
	this->cols = cols;
	this->rows = rows;
	this->x = new double[cols];
	this->y = new double[cols];

	this->measures = 0;
	this->measuring = false;

	this->elapsed_time = 0.0;

	srand(time(NULL));

	for (int i = 0; i < cols; i++) {
		double integer_part = rand();
		double decimal_part = rand() / RAND_MAX;
		this->x[i] = integer_part + decimal_part;
	}
}

int Matrix::getnz() {
	return this->nz;
}

double * Matrix::getX() {
	return this->x;
}

int Matrix::getCols() {
	return this->cols;
}

unsigned long long Matrix::getGigaFlops() {
	return 2.0 * (unsigned long long)this->nz / ((double)this->elapsed_time / (unsigned long long)this->measures) / 1000000.0;
}

void Matrix::printElapsedTime() {
	std::cout << " (" << ((double)this->elapsed_time / (double)this->measures) << " ms " << this->measures << " measures) ";
}

void Matrix::resetResults() {
	//delete [] this->y;
	//this->y = new double[cols];
	this->measures = 0;
	this->measuring = false;
	this->elapsed_time = 0.0;
}

void Matrix::trackTime() {
	if (this->measuring == false) {
		this->start = omp_get_wtime();
		this->measuring = true;
	} else {
		this->done = omp_get_wtime();
		this->elapsed_time += done - start;
		this->measures++;
		this->measuring = false;
	}
}