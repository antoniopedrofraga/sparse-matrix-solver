#include <stdio.h>
#include <iostream>

#include "matrix/matrix.h"
#include "matrix/csr.h"
#include "matrix/ellpack.h"
#include "io/iomanager.h"
#include "utils/utils.h"
 
__global__ void solveCSR(CSR * csr) {
	int i = threadIdx.x;
	double t = 0.0;
	for (int j = csr->irp[i]; j < csr->irp[i + 1] - 1; j++) {
			t += csr->as[j] * csr->x[csr->ja[j]];
	}
	csr->y[i] = t;
}

__global__ void solveEllpack(Ellpack * ellpack) {
	int i = threadIdx.x;
	double t = 0.0;
	for (int j = 0; j < ellpack->maxnz; j++) {
		t += ellpack->as[i][j] * ellpack->x[ellpack->ja[i][j]];
	}
	ellpack->y[i] = t;
}

void solveCuda(IOmanager * io, std::string path, CSR * &csr, Ellpack * &ellpack) {
	
	const int m = csr->getCols();
	const int csize = sizeof(CSR);
	const int esize = sizeof(Ellpack);
	
	CSR * csr_c;
	Ellpack * ellpack_c;
	float elapsedtime = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&csr_c, csize);
	cudaMalloc((void**)&ellpack_c, esize);
	cudaMemcpy(csr_c, csr, csize, cudaMemcpyHostToDevice); 
	cudaMemcpy(ellpack_c, ellpack, esize, cudaMemcpyHostToDevice); 
	
	for (int k = 0; k < NR_RUNS; ++k) {
		cudaEventRecord(start);
		solveCSR<<<1, m>>>(csr_c);
		cudaThreadSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedtime, start, stop);
		csr->addElapsedTime(elapsedtime);

		cudaEventRecord(start);
		solveEllpack<<<1, m>>>(ellpack_c);
		cudaThreadSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedtime, start, stop);
		ellpack->addElapsedTime(elapsedtime);
	}
	
	io->exportResults(CUDA, path, csr, ellpack);

	cudaFree(csr_c);
	cudaFree(ellpack_c);
}
