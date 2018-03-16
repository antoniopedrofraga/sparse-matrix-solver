#include "matrix.h"
#include <time.h>
#include <iostream>

Matrix::Matrix(int cols, int rows, int nz) {
	this->nz = nz;
	this->cols = cols;
	this->rows = rows;
	this->x = new double[cols];
	this->y = new double[cols];

	this->measures = 0;
	this->measuring = false;

	this->start = 0;
	this->elapsed_time = 0;
	
	this->flops = -1;

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

int Matrix::getFlops() {
	return this->flops;
}

void Matrix::trackTime() {
	if (!this->measuring) {
		this->start = clock();
		this->measuring = true;
	} else {
		this->measures++;
		this->elapsed_time += (clock() - this->start) / CLOCKS_PER_SEC;
		this->flops = this->elapsed_time == 0 ? 0 : 2.0 * (double)this->nz / (this->elapsed_time / this->measures);
		this->measuring = false;

		std::cout << " (" << this->elapsed_time << " seconds) ";
	}
}
