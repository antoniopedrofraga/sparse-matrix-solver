#include "ellpack.h"

Ellpack::Ellpack(int cols, int rows, int maxnz) : Matrix(cols, rows) {
	this->maxnz = maxnz;
	this->ja = new int * [cols];
	this->as = new double * [cols];
	
	std::fill(&this->ja[0], &this->ja[cols], new int[maxnz]);
	std::fill(&this->as[0], &this->as[cols], new double[maxnz]);
	
	for (int i = 0; i < cols; i++) {
		std::fill(&this->ja[i][0], &this->ja[i][maxnz], 0);
		std::fill(&this->as[i][0], &this->as[i][maxnz], 0.0);
	}
};
