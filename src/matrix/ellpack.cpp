#include "ellpack.h"

Ellpack::Ellpack(int cols, int rows, int maxnz) : Matrix(cols, rows, maxnz * cols) {
	this->maxnz = maxnz;
	this->ja = new int * [cols];
	this->pointer = new int[cols];
	this->as = new double * [cols];
	
	std::fill(&this->ja[0], &this->ja[cols], new int[maxnz]);
	std::fill(&this->as[0], &this->as[cols], new double[maxnz]);
	std::fill(&this->pointer[0], &this->pointer[cols], 0);
	
	for (int i = 0; i < cols; ++i) {
		std::fill(&this->ja[i][0], &this->ja[i][maxnz], -1);
		std::fill(&this->as[i][0], &this->as[i][maxnz], 0.0);
	}
};

Ellpack::~Ellpack() {
	delete [] this->x;
	delete [] this->y;
	delete [] this->pointer;
	delete [] this->ja;
	delete [] this->as;
};

void Ellpack::addElement(int col_index, int row_index, double value) {
	int p = this->pointer[col_index];
	this->ja[col_index][p] = row_index;
	this->as[col_index][p] = value;
	this->pointer[col_index]++;
}

int Ellpack::getmaxnz() {
	return this->maxnz;
}

int ** Ellpack::getja() {
	return this->ja;
}

int * Ellpack::getpointers() {
	return this->pointer;
}

double ** Ellpack::getas() {
	return this->as;
}
