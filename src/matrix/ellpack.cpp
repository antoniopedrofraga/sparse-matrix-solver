#include "ellpack.h"
#include <iostream>
#include <climits>

Ellpack::Ellpack(int cols, int rows, int maxnz, int nz) : Matrix(cols, rows, nz) {
	this->maxnz = maxnz;
	this->ja = alloc2d(rows, maxnz);
	this->pointer = new int[rows];
	this->as = alloc2dDouble(rows, maxnz);
	/*
	for (int i = 0; i < rows; ++i) {
		this->ja[i] = new int[maxnz];
		this->as[i] = new double[maxnz];
	}*/
	
	std::fill(&this->pointer[0], &this->pointer[rows], 0);
	
	for (int i = 0; i < rows; ++i) {
		std::fill(&this->ja[i][0], &this->ja[i][maxnz], 0);
		std::fill(&this->as[i][0], &this->as[i][maxnz], 0.0);
	}
};

Ellpack::~Ellpack() {
	delete [] this->x;
	delete [] this->y;
	delete [] this->pointer;
	delete [] this->ja;
	delete [] this->as;
	delete [] this->onedas;
	delete [] this->onedja;
};

void Ellpack::addElement(int row_index, int col_index, double value) {
	int p = this->pointer[row_index];
	this->ja[row_index][p] = col_index;
	this->as[row_index][p] = value;
	this->pointer[row_index]++;
}

int Ellpack::getmaxnz() {
	return this->maxnz;
}

int ** Ellpack::getja() {
	return this->ja;
}

int * Ellpack::get1Dja() {
	int m = getRows();
	int mnz = getmaxnz();
	onedja = new int[m * mnz];
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < mnz; ++j) {
			onedja[i * mnz + j] = ja[i][j];
		}
	}
	return onedja;
}

bool Ellpack::toLargeForCUDA() {
	__int128 m = getRows();
	__int128 mnz = getmaxnz();
	return m * mnz > INT_MAX;
}

int * Ellpack::getpointers() {
	return this->pointer;
}

double ** Ellpack::getas() {
	return this->as;
}

double * Ellpack::get1Das() {
	int m = getRows();
	int mnz = getmaxnz();
	onedas = new double[m * mnz];
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < mnz; ++j) {
			onedas[i * mnz + j] = as[i][j];
		}
	}
	return onedas;
}

void Ellpack::print() {
	std::cout << "M = " << getRows() << std::endl;
	std::cout << "N = " << getCols() << std::endl;
	std::cout << "MAXNZ = " << maxnz << std::endl;
	std::cout << "JA = " << std::endl;
	for (int i = 0; i < getRows(); ++i) {
		for (int j = 0; j < maxnz; ++j) {
			std::cout << ja[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "AS = " << std::endl;
	for (int i = 0; i < getRows(); ++i) {
		for (int j = 0; j < maxnz; ++j) {
			std::cout << as[i][j] << " ";
		}
		std::cout << std::endl;
	}
}
