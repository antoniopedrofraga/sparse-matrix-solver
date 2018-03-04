#include "solver.h"
#include <vector>
#include <omp.h>

Solver::Solver(std::pair<CSR*, Ellpack*> &matrices) {
	this->csr = matrices.first;
	this->ellpack = matrices.second;
}

void Solver::sequentialCSR() {
	int m = csr->getCols();
	std::vector<int> irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();
	double * y = new double[m];

	for (int i = 0; i < m; i++) {
		double t = 0.0;
		for (size_t k = 0; k < irp.size(); k++) {
			int j = irp[k];
			t += as[j] * x[ja[j]];
		}
		y[i] = t;	
	}
}

void Solver::sequentialEllpack() {
	int m = ellpack->getCols();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();
	double * y = new double[m];

	for (int i = 0; i < m; i++) {
		double t = 0.0;
		for (int j = 0; j < maxnz; j++) {
			t += as[i][j] * x[ja[i][j]];
		}
		y[i] = t;
	}
}


void Solver::openmpCSR() {
	int m = csr->getCols();
	std::vector<int> irp = csr->getirp();
	int * ja = csr->getja();
	double * as = csr->getas();
	double * x = csr->getX();
	double * y = new double[m];
	
	int i, j; size_t k;
	double t = 0.0;
	#pragma omp parallel for private(i, j, k) reduction(+:t)
	for (i = 0; i < m; i++) {
		t = 0.0;
		for (k = 0; k < irp.size(); k++) {
			j = irp[k];
			t += as[j] * x[ja[j]];
		}
		y[i] = t;	
	}
}

void Solver::openmpEllpack() {
	int m = ellpack->getCols();
	int maxnz = ellpack->getmaxnz();
	int ** ja = ellpack->getja();
	double ** as = ellpack->getas();
	double * x = ellpack->getX();
	double * y = new double[m];

	int i, j;
	double t = 0.0;
	#pragma omp parallel for private(i, j) reduction(+:t)
	for (i = 0; i < m; i++) {
		t = 0.0;
		for (j = 0; j < maxnz; j++) {
			t += as[i][j] * x[ja[i][j]];
		}
		y[i] = t;
	}
}

void Solver::sequential() {
	sequentialCSR();
	sequentialEllpack();
}

void Solver::cuda() {
	
}

void Solver::openMP() {
	openmpCSR();
	openmpEllpack();
}
