#include <iostream>
#include <string>

#include "matrix/csr.h"
#include "matrix/ellpack.h"
#include "io/iomanager.h"
#include "matrix/matrix.h"

void solveCSR(CSR * csr) {
	int m = csr->getCols();
	int * irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();

	for (int i = 0; i < m; i++) {
		double t = 0.0;
		for (size_t k = 0; k < csr->irp_size; k++) {
			int j = irp[k];
			t += as[j] * x[ja[j]];
		}
		csr->y[i] = t;	
	}
}

void solveEllpack(Ellpack * ellpack) {
	int m = ellpack->getCols();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();

	for (int i = 0; i < m; i++) {
		double t = 0.0;
		for (int j = 0; j < maxnz; j++) {
			t += as[i][j] * x[ja[i][j]];
		}
		ellpack->y[i] = t;
	}
}

int main(int argc, char ** argv) {	
	IOmanager * io = new IOmanager();
	std::string path = io->parseArguments(argc, argv);
	std::pair<CSR*, Ellpack*> matrices = io->readFile(path);

	solveCSR(matrices.first);
	solveEllpack(matrices.second);
}



