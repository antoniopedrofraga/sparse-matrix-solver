#include "matrix.h"
#include <time.h>
#include <iostream>
#include <chrono>
#include <limits>

Matrix::Matrix(int cols, int rows, int nz) {
	this->nz = nz;
	this->rows = rows;
	this->cols = cols;
	this->x = new double[rows];
	this->y = new double[rows];
	
	this->measures = 0;
	this->measuring = false;

	this->elapsed_time = 0.0;
	this->omp_times_threads = new std::pair<int, double>[NUM_THREADS];

	std::fill(&this->y[0], &this->y[0], 0.0);
	srand(time(NULL));

	for (int i = 0; i < NUM_THREADS; ++i) {
		omp_times_threads[i] = std::make_pair(0, 0.0);
	}
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

int Matrix::getRows() {
	return this->rows;
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
	delete [] this->omp_times_threads;

	this->omp_times_threads = new std::pair<int, double>[NUM_THREADS];
	this->y = new double[cols];
	std::fill(&this->y[0], &this->y[0], 0.0);

	for (int i = 0; i < NUM_THREADS; ++i) {
		omp_times_threads[i] = std::make_pair(0, 0.0);
	}

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