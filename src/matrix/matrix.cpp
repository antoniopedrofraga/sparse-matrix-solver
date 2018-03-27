#include "matrix.h"
#include <time.h>
#include <iostream>
#include <chrono>
#include <limits>

Matrix::Matrix(int cols, int rows, int nz) {
	this->nz = nz;
	this->cols = cols;
	this->rows = rows;
	this->x = new double[cols];
	this->y = new double[cols];
	
	this->measures = 0;
	this->measuring = false;

	this->elapsed_time = 0.0;
	this->omp_times_threads = new std::pair<int, double>[NUM_THREADS];

	std::fill(&omp_times_threads[0], &omp_times_threads[NUM_THREADS], std::make_pair(0, 0.0));
	std::fill(&y[0], &y[cols], 0.0);
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

unsigned long long Matrix::getMegaFlops() {
	return 2.0 * (unsigned long long)this->nz / ((double)this->elapsed_time / (unsigned long long)this->measures) / 1000.0;
}

unsigned long long Matrix::getMegaFlops(int i) {
	return 2.0 * (unsigned long long)this->nz / ((double)this->omp_times_threads[i].second / (unsigned long long)this->omp_times_threads[i].first) / 1000.0;
}

void Matrix::printElapsedTime() {
	std::cout << " (" << ((double)this->elapsed_time / (double)this->measures) << " ms " << this->measures << " measures) ";
}

void Matrix::resetResults() {
	delete [] this->y;
	this->y = new double[cols];
	std::fill(&y[0], &y[cols], 0.0);

	this->measures = 0;
	this->measuring = false;
	this->elapsed_time = 0.0;
}

void Matrix::trackTime() {
	if (this->measuring == false) {
		this->start = omp_get_wtime() * 1000;
		this->measuring = true;
	} else {
		this->done = omp_get_wtime() * 1000;
		this->elapsed_time += done - start;
		this->measures++;
		this->measuring = false;
	}
}

void Matrix::trackTimeOMP(int num_threads) {
	if (this->measuring == false) {
		this->start = omp_get_wtime() * 1000;
		this->measuring = true;
	} else {
		int pos = num_threads - MIN_THREADS;
		this->done = omp_get_wtime() * 1000;
		this->omp_times_threads[pos].second += done - start;
		this->omp_times_threads[pos].first++;
		this->measuring = false;

		if (this->omp_times_threads[pos].first == NR_RUNS) {
			if (num_threads == MIN_THREADS) {
				this->elapsed_time = this->omp_times_threads[pos].second;
				this->measures = NR_RUNS;
			} else {
				if (this->elapsed_time / this->measures > this->omp_times_threads[pos].second / this->omp_times_threads[pos].first) {
					this->elapsed_time = this->omp_times_threads[pos].second;
				}
			}
		}
	}
}