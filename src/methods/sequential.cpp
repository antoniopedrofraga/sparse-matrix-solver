#include <iostream>
#include <string>

#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../io/iomanager.h"
#include "../matrix/matrix.h"

void sequentialCSR(CSR * &csr) {
	int m = csr->getCols();
	int * irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();

	for (int k = 0; k < NR_RUNS; ++k) {
		csr->trackTime();
		for (int i = 0; i < m; ++i) {
			for (int j = irp[i]; j < irp[i + 1]; ++j) {
				if (j < 0 || j >= csr->getnz()) {
					std::cout << "From " << j << " to " << irp[i + 1] << std::endl;
				}
				if (ja[j] < 0 || ja[j] >= csr->getnz()) {
					std::cout << "ja[j] = " << ja[j] << " of " << csr->getnz() << std::endl;
				}
				csr->y[i] += as[j] * x[ja[j]];
			}
		}
		csr->trackTime();
	}
}

void sequentialEllpack(Ellpack * &ellpack) {
	int m = ellpack->getCols();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();

	for (int k = 0; k < NR_RUNS; ++k) {
		ellpack->trackTime();
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < maxnz; ++j) {
				ellpack->y[i] += as[i][j] * x[ja[i][j]];
			}
		}
		ellpack->trackTime();
	}
}


