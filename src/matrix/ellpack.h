#ifndef ELLPACK_H
#define ELLPACK_H

#include "matrix.h"

class Ellpack : public Matrix {
	int maxnz, ** ja;
	double ** as;
public:
	Ellpack(int cols, int rows, int maxnz);
};

#endif
