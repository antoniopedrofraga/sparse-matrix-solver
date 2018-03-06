#include <stdio.h>
#include <iostream>

#include "matrix/csr.h"
#include "matrix/ellpack.h"
#include "io/iomanager.h"
#include "matrix/matrix.h"
 
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

void solveCuda(CSR * csr, Ellpack * ellpack) {
	
	const int m = csr->getCols();
	const int csize = sizeof(CSR);
	const int esize = sizeof(Ellpack);
	
	CSR * csr_c;
	Ellpack * ellpack_c;

	cudaMalloc((void**)&csr_c, csize);
	cudaMalloc((void**)&ellpack_c, esize);
	cudaMemcpy(csr_c, csr, csize, cudaMemcpyHostToDevice); 
	cudaMemcpy(ellpack_c, ellpack, esize, cudaMemcpyHostToDevice); 
	
	solveCSR<<<1, m>>>(csr_c);
	solveEllpack<<<1, m>>>(ellpack_c);
	
	cudaMemcpy(csr, csr_c, csize, cudaMemcpyDeviceToHost); 
	cudaMemcpy(ellpack, ellpack_c, esize, cudaMemcpyDeviceToHost); 
	cudaFree(csr_c);
	cudaFree(ellpack_c);

	/*char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
	char *ad;
	int *bd;
	const int csize = N * sizeof(char);
	const int isize = N * sizeof(int);
 
	printf("%s", a);
 
	cudaMalloc((void**)&ad, csize); 
	cudaMalloc((void**)&bd, isize); 
	cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice); 
	cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice); 
  
	dim3 dimGrid(25, 25);
	dim3 dimBlock(10, 10);
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost); 
	cudaFree(ad);
	cudaFree(bd);

	std::cout << a << std::endl;*/
}
