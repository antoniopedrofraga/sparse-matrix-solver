#include <stdio.h>
#include <iostream>

#include "../matrix/matrix.h"
#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../io/iomanager.h"
#include "../utils/utils.h"
 
__global__ void solveCSR(CSR * csr) {
	int i = threadIdx.x;
	for (int j = csr->irp[i]; j < csr->irp[i + 1] - 1; ++j) {
			csr->y[i] += csr->as[j] * csr->x[csr->ja[j]];
	}
}

__global__ void solveEllpack(Ellpack * ellpack) {
	int i = threadIdx.x;
	for (int j = 0; j < ellpack->maxnz; ++j) {
		ellpack->y[i] += ellpack->as[i][j] * ellpack->x[ellpack->ja[i][j]];
	}
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
	
	for (int k = 0; k < NR_RUNS; ++k) {
		csr->trackTime();
		solveCSR<<<1, m>>>(csr_c);
		cudaThreadSynchronize();
		csr->trackTime();
		
		ellpack->trackTime();
		solveEllpack<<<1, m>>>(ellpack_c);
		cudaThreadSynchronize();
		ellpack->trackTime();
	}
	
	io->exportResults(CUDA, path, csr, ellpack);

	cudaFree(csr_c);
	cudaFree(ellpack_c);
}
