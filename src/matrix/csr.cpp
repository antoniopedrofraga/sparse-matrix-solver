#include "csr.h"
#include <iostream>

CSR::CSR(int cols, int rows, int nz) : Matrix(cols, rows, nz) {
	try {
		this->ja = new (std::nothrow) int[nz];
		this->as = new (std::nothrow)double[nz];
		this->irp = new (std::nothrow) int[cols + 1];
	} catch (std::bad_alloc& ba) {
		this->rows = 0;
		this->cols = 0;
		this->fits_in_memory = false;
	}

	this->element_index = 0;

	this->cuda_times_csr = std::make_pair(0.0, 0.0);
	
	this->irp_size = 0;

	if (this->fits_in_memory) {
		std::fill(&this->irp[0], &this->irp[cols + 1], -1);
		std::fill(&this->ja[0], &this->ja[nz], -1);
		std::fill(&this->as[0], &this->as[nz], 0.0);
		this->initVectors();
	}
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

void CSR::print() {
	std::cout << "M = " << getRows() << std::endl;
	std::cout << "N = " << getCols() << std::endl;
	std::cout << "IRP = ";
	for (int i = 0; i < getCols() + 1; ++i) {
		std::cout << irp[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "JA = ";
	for (int i = 0; i < getnz(); ++i) {
		std::cout << ja[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "AS = ";
	for (int i = 0; i < getnz(); ++i) {
		std::cout << as[i] << " ";
	}
	std::cout << std::endl;
}

unsigned long long CSR::getScalarMegaFlops() {
	return 2.0 * (unsigned long long)getnz() / ((double)this->cuda_times_csr.first / (double)this->measures) / 1000.0;
}

unsigned long long CSR::getVecMinMegaFlops() {
	return 2.0 * (unsigned long long)getnz() / ((double)this->cuda_times_csr.second / (unsigned long long)this->measures) / 1000.0;
}

void CSR::trackCSRTime(int method) {
	if (this->measuring == false) {
		this->start = omp_get_wtime() * 1000;
		this->measuring = true;
	} else {
		this->done = omp_get_wtime() * 1000;
		if (method == SCALAR) {
			this->cuda_times_csr.first += done - start;
			this->measures++;
		} else if (method == VECTOR_MINING) {
			this->cuda_times_csr.second += done - start;
		}
		this->measuring = false;
	}
}

void CSR::printElapsedCUDATime() {
	std::cout << " SCALAR(" << ((double)this->cuda_times_csr.first / (double)this->measures) << " ms " << this->measures << " measures) ";
	std::cout << " VM(" << ((double)this->cuda_times_csr.second / (double)this->measures) << " ms " << this->measures << " measures) ";
}
