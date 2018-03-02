#include "matrix.h"
#include <random>

Matrix::Matrix(int cols, int rows) {
	this->cols = cols;
	this->rows = rows;
	this->x = new double[cols];

	static std::default_random_engine eng;
	static std::uniform_real_distribution<> dis(0.0001, 99999.9999);

	for (int i = 0; i < cols; i++) {
		this->x[i] = dis(eng);
	}
}
