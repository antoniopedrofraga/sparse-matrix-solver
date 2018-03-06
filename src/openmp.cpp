#include <iostream>
#include <string>
#include <omp.h>

#include "matrix/csr.h"
#include "matrix/ellpack.h"
#include "io/iomanager.h"
#include "matrix/matrix.h"

void openmpCSR(CSR * csr) {
	int m = csr->getCols();
	int * irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();
	
	int i, j; size_t k;
	double t = 0.0;
	#pragma omp parallel for private(i, j, k) reduction(+:t)
	for (i = 0; i < m; i++) {
		t = 0.0;
		for (k = 0; k < csr->irp_size; k++) {
			j = irp[k];
			t += as[j] * x[ja[j]];
		}
		csr->y[i] = t;	
	}
}

void openmpEllpack(Ellpack * ellpack) {
	int m = ellpack->getCols();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();

	int i, j;
	double t = 0.0;
	#pragma omp parallel for private(i, j) reduction(+:t)
	for (i = 0; i < m; i++) {
		t = 0.0;
		for (j = 0; j < maxnz; j++) {
			t += as[i][j] * x[ja[i][j]];
		}
		ellpack->y[i] = t;
	}
}

int main(int argc, char ** argv) {	
	IOmanager * io = new IOmanager();
	std::string path = io->parseArguments(argc, argv);
	std::pair<CSR*, Ellpack*> matrices = io->readFile(path);

	openmpCSR(matrices.first);
	openmpEllpack(matrices.second);
}


