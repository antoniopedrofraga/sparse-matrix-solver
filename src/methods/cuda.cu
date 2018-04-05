#include <stdio.h>
#include <iostream>

#include "../matrix/matrix.h"
#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../io/iomanager.h"
#include "../utils/utils.h"


int thread_block = 512;


void cudaCheckError() {
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
		exit(0);
	}
}

__global__ void solveCSR(int * m, int * irp, int * ja, double * as, double * x, double * y) {
	extern __shared__ double sdata[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	/*int tid = threadIdx.x;

	int warp = i / 32; 
	int lane = i & (32 - 1); 

	int row = warp;
	sdata[tid] = 0;

	if (row < *m) {
		for (int j = irp[row] + lane ; j < irp[row + 1]; j += 32)
			sdata[tid] += as[j] * x[ja[j]];

		if (lane < 16) sdata[tid] += sdata[tid + 16];
		if (lane < 8) sdata[tid] += sdata[tid + 8];
		if (lane < 4) sdata[tid] += sdata[tid + 4];
		if (lane < 2) sdata[tid] += sdata[tid + 2];
		if (lane < 1) sdata[tid] += sdata[tid + 1];

		if (lane == 0)
			y[row] += sdata[tid];
	}*/
	if (i < *m) {
		double temp = 0.0;
		for (int j = irp[i]; j < irp[i + 1]; ++j) {
			temp += as[j] * x[ja[j]];
		}
		y[i] = temp;
	}
	__syncthreads();
}

__global__ void solveEllpack(int * m, int * ja, double * as, double * x, double * y, int * maxnz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int rows = *m, maximum_nz = *maxnz;
	if (i < rows) {
		double temp = 0.0;
		for (int j = 0; j < *maxnz; ++j) {
			temp += as[i * maximum_nz + j] * x[ja[i * maximum_nz + j]];
		}
		y[i] = temp;
	}
	__syncthreads();
}

void allocateCSR(CSR * &csr, int * &irp, int * &ja, double * &as, double * &x, double *&y, int &m, int &n) {
	int nz = csr->getnz();

	cudaMalloc((void**)&irp, sizeof(int) * (nz + 1));
	cudaMalloc((void**)&ja, sizeof(int) * nz);
	cudaMalloc((void**)&as, sizeof(double) * nz);
	cudaMalloc((void**)&x, sizeof(double) * n);
	cudaMalloc((void**)&y, sizeof(double) * n);
	cudaMemcpy(irp, csr->getirp(), sizeof(int) * (nz + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ja, csr->getja(), sizeof(int) * nz, cudaMemcpyHostToDevice);
	cudaMemcpy(as, csr->getas(), sizeof(double) * nz, cudaMemcpyHostToDevice);
	cudaMemcpy(x, csr->getX(), sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(y, csr->y, sizeof(double) * n, cudaMemcpyHostToDevice);
}

void allocateEllpack(Ellpack * &ellpack, int * &ja, double * &as, double * &x, double * &y, int * &maxnz, int &m, int &n) {
	int host_maxnz = ellpack->getmaxnz();
	int * host_ja = ellpack->get1Dja();
	double * host_as = ellpack->get1Das();

	cudaMalloc((void**)&ja, sizeof(int) * m * host_maxnz);
	cudaMalloc((void**)&as,  sizeof(double) * m * host_maxnz);
	cudaMalloc((void**)&x, sizeof(double) * m);
	cudaMalloc((void**)&y, sizeof(double) * m);
	cudaMalloc((void**)&maxnz, sizeof(int));
	cudaMemcpy(ja, host_ja, sizeof(int) * m * host_maxnz, cudaMemcpyHostToDevice);
	cudaMemcpy(as, host_as, sizeof(double) * m * host_maxnz, cudaMemcpyHostToDevice);
	cudaMemcpy(x, ellpack->getX(), sizeof(double) * m, cudaMemcpyHostToDevice);
	cudaMemcpy(y, ellpack->y, sizeof(double) * m, cudaMemcpyHostToDevice);
	cudaMemcpy(maxnz, &host_maxnz, sizeof(int), cudaMemcpyHostToDevice);
}

void collectResults(CSR * &csr, Ellpack * &ellpack, double * &csr_y, double * &ellpack_y, int &n) {
	cudaMemcpy(csr->y, csr_y, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(ellpack->y, ellpack_y, sizeof(double) * n, cudaMemcpyDeviceToHost);
}


void solveCuda(IOmanager * io, std::string path, CSR * &csr, Ellpack * &ellpack) {
	
	int m = csr->getRows();
	int n = csr->getCols();
	const int shmem_size = thread_block * sizeof(double);

	int n_blocks = m / thread_block;

	if (m % thread_block > 0.0) {
		n_blocks++;
	}

	int * csr_irp, * csr_ja, * ellpack_ja, * maxnz, * rows;
	double * csr_as, * csr_x, * csr_y, * ellpack_as, * ellpack_x, * ellpack_y;
	
	allocateCSR(csr, csr_irp, csr_ja, csr_as, csr_x, csr_y, m, n);
	allocateEllpack(ellpack, ellpack_ja, ellpack_as, ellpack_x, ellpack_y, maxnz, m, n);

	cudaMalloc((void**)&rows, sizeof(int));
	cudaMemcpy(rows, &m, sizeof(int), cudaMemcpyHostToDevice);

	
	for (int k = 0; k < NR_RUNS; ++k) {
		csr->trackTime();
		solveCSR<<<n_blocks, thread_block, shmem_size>>>(rows, csr_irp, csr_ja, csr_as, csr_x, csr_y);
		csr->trackTime();
		cudaMemset(csr_y, 0, sizeof(double) * m);
		cudaCheckError();
		
		ellpack->trackTime();
		solveEllpack<<<n_blocks, thread_block>>>(rows, ellpack_ja, ellpack_as, ellpack_x, ellpack_y, maxnz);
		ellpack->trackTime();
		cudaMemset(ellpack_y, 0, sizeof(double) * m);
		cudaCheckError();
	}

	collectResults(csr, ellpack, csr_y, ellpack_y, n);
	cudaCheckError();
	
	io->exportResults(CUDA, path, csr, ellpack);
}
