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
	for (size_t k = 0; k < csr->irp_size; k++) {
		int j = csr->irp[k];
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

	cudaMalloc((void**)&csr_c, csize);
	cudaMalloc((void**)&ellpack_c, esize);
	cudaMemcpy(csr_c, csr, csize, cudaMemcpyHostToDevice); 
	cudaMemcpy(ellpack_c, ellpack, esize, cudaMemcpyHostToDevice); 
	
	csr->trackTime();
	solveCSR<<<1, m>>>(csr_c);
	cudaThreadSynchronize();
	csr->trackTime();

	ellpack->trackTime();
	solveEllpack<<<1, m>>>(ellpack_c);
	cudaThreadSynchronize();
	ellpack->trackTime();
	
	cudaMemcpy(csr, csr_c, csize, cudaMemcpyDeviceToHost); 
	cudaMemcpy(ellpack, ellpack_c, esize, cudaMemcpyDeviceToHost); 
	cudaFree(csr_c);
	cudaFree(ellpack_c);

	io->exportResults(cuda, path, csr, ellpack);
}
