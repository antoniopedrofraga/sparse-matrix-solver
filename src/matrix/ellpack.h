#ifndef ELLPACK_H
#define ELLPACK_H

#include "matrix.h"

class Ellpack : public Matrix {
public:
	int maxnz, ** ja, * pointer;
	double ** as;
	
	Ellpack(int cols, int rows, int maxnz);
	void addElement(int col_index, int row_index, double value);


	int getmaxnz();
	int ** getja();
	int * getpointers();
	double ** getas();
};

#endif
