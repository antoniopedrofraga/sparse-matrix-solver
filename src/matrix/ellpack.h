#ifndef ELLPACK_H
#define ELLPACK_H

#include "matrix.h"

class Ellpack : public Matrix {
public:
	int maxnz, ** ja, * pointer, * onedja;
	double ** as, * onedas;
	
	Ellpack(int cols, int rows, int maxnz, int nz);
	~Ellpack();
	
	void addElement(int col_index, int row_index, double value);

	int getmaxnz();
	int ** getja();
	int * get1Dja();
	int * getpointers();
	double ** getas();
	double * get1Das();
	void print();
};

#endif
