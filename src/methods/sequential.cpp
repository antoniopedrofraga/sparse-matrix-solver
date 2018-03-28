#include <iostream>
#include <string>
#include <limits>

#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../io/iomanager.h"
#include "../matrix/matrix.h"

void pause() {
    std::cin.clear();
    std::cin.ignore(numeric_limits<streamsize>::max(), '\n');
    std::string dummy;
    std::cout << "Press any key to continue . . .";
    std::getline(std::cin, dummy);
}

void sequentialCSR(CSR * &csr) {
	int m = csr->getRows();
	int * irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();
	double temp = 0.0;
	for (int k = 0; k < NR_RUNS; ++k) {
		csr->trackTime();
		for (int i = 0; i < m; ++i) {
			temp = 0.0;
			for (int j = irp[i]; j < irp[i + 1]; ++j) {
				temp += as[j] * x[ja[j]];
			}
			csr->y[i] = temp;
		}
		csr->trackTime();
	}
}

void sequentialEllpack(Ellpack * &ellpack) {
	int m = ellpack->getRows();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();

	double temp = 0.0;
	for (int k = 0; k < NR_RUNS; ++k) {
		ellpack->trackTime();
		for (int i = 0; i < m; ++i) {
			temp = 0.0;
			for (int j = 0; j < maxnz; ++j) {
				temp += as[i][j] * x[ja[i][j]];
			}
			ellpack->y[i] = temp;
		}
		ellpack->trackTime();
	}
}

