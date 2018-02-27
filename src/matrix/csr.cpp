#include "csr.h"

CSR::CSR(int cols, int rows, int nz) : Matrix(cols, rows) {
	this->nz = nz;
	this->ja = new int[nz];
	this->as = new double[nz];

	std::fill(&this->ja[0], &this->ja[nz], 0);
	std::fill(&this->as[0], &this->as[nz], 0.0);
};
