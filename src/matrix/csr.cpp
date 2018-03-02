#include "csr.h"

CSR::CSR(int cols, int rows, int nz) : Matrix(cols, rows) {
	this->nz = nz;
	this->ja = new int[nz];
	this->as = new double[nz];
	this->element_index = 0;

	std::fill(&this->ja[0], &this->ja[nz], 0);
	std::fill(&this->as[0], &this->as[nz], 0.0);
};

void CSR::addPointer(int pointer) {
	this->irp.push_back(pointer);
}

void CSR::addElement(int col_index, double value) {
	this->ja[element_index] = col_index;
	this->as[element_index] = value;
	this->element_index++;
}

int * CSR::getja() {
	return this->ja;
}

double * CSR::getas() {
	return this->as;
}

std::vector<int> CSR::getirp() {
	return this->irp;
}