#include <iostream>
#include <string>
#include <omp.h>

#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../io/iomanager.h"
#include "../matrix/matrix.h"
#include "../utils/utils.h"

void openmpCSR(CSR * &csr) {
	int m = csr->getCols();
	int * irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();
	double * y = csr->y;
	
	int i;
	for (int t = MIN_THREADS; t <= MAX_THREADS; ++t) {
		for (int k = 0; k <= NR_RUNS; ++k) {
			if (k != 0) csr->trackTimeOMP(t);
			#pragma omp parallel for private(i) schedule(static) num_threads(t)
			for (i = 0; i < m; ++i) {
				for (int j = irp[i]; j < irp[i + 1] - 1; ++j) {
					y[i] += (double)as[j] * (double)x[ja[j]];
				}
			}
			if (k != 0) csr->trackTimeOMP(t);
		}
	}
}

void openmpEllpack(Ellpack * &ellpack) {
	int m = ellpack->getCols();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();

	int i;
	for (int t = MIN_THREADS; t <= MAX_THREADS; ++t) {
		for (int k = 0; k <= NR_RUNS; ++k) {
			if (k != 0) ellpack->trackTimeOMP(t);
			#pragma omp parallel for private(i) schedule(static) num_threads(t)
			for (i = 0; i < m; ++i) {
				for (int j = 0; j < maxnz; ++j) {
					ellpack->y[i] += as[i][j] * x[ja[i][j]];
				}
			}
			if (k != 0) ellpack->trackTimeOMP(t);
		}
	}
}


