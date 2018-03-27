#include "csr.h"
#include <iostream>

CSR::CSR(int cols, int rows, int nz) : Matrix(cols, rows, nz) {
	this->ja = new int[nz];
	this->as = new double[nz];
	this->irp = new int[cols + 1];
	this->element_index = 0;
	
	this->irp_size = 0;

	std::fill(&this->irp[0], &this->irp[cols + 1], -1);
	std::fill(&this->ja[0], &this->ja[nz], 0);
	std::fill(&this->as[0], &this->as[nz], 0.0);
};

CSR::~CSR() {
	delete [] this->x;
	delete [] this->y;
	delete [] this->ja;
	delete [] this->as;
	delete [] this->irp;
};

void CSR::addPointer(int pointer) {
	this->irp[this->irp_size] = pointer;
	this->irp_size++;
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

int * CSR::getirp() {
	return this->irp;
}
