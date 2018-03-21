#include <iostream>
#include <string>
#include <omp.h>

#include "matrix/csr.h"
#include "matrix/ellpack.h"
#include "io/iomanager.h"
#include "matrix/matrix.h"
#include "utils/utils.h"

void openmpCSR(CSR * &csr) {
	int m = csr->getCols();
	int * irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();
	
	int i;

	for (int k = 0; k < NR_RUNS + 1; ++k) {
		if (k != 0) csr->trackTime();
		#pragma omp parallel for private(i) schedule(static) num_threads(4)
		for (i = 0; i < m; ++i) {
			for (int j = irp[i]; j < irp[i + 1]; ++j) {
				csr->y[i] += as[j] * x[ja[j]];
			}
		}
		if (k != 0) csr->trackTime();
	}
}

void openmpEllpack(Ellpack * &ellpack) {
	int m = ellpack->getCols();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();

	int i;
	for (int k = 0; k < NR_RUNS + 1; ++k) {
		if (k != 0) ellpack->trackTime();
		#pragma omp parallel for private(i) schedule(static) num_threads(4)
		for (i = 0; i < m; ++i) {
			for (int j = 0; j < maxnz; ++j) {
				ellpack->y[i] += as[i][j] * x[ja[i][j]];
			}
		}
		if (k != 0) ellpack->trackTime();
	}
}

int main(int argc, char ** argv) {	
	IOmanager * io = new IOmanager();
	std::string path = io->parseArguments(argc, argv);
	std::pair<CSR*, Ellpack*> matrices = io->readFile(path);

	openmpCSR(matrices.first);
	openmpEllpack(matrices.second);

	io->exportResults(OPENMP, path, matrices.first, matrices.second);
}


