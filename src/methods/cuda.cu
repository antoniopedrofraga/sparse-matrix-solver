#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_timer.h>


#include "../matrix/matrix.h"
#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../io/iomanager.h"
#include "../utils/utils.h"


unsigned int vm_thr_block = 512, scalar_thr_block = 512, n_blocks_vm, n_blocks_scalar;


void cudaCheckError(int line) {
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, line, cudaGetErrorString(e));
	}
}



__global__ void scalarCSR(int * m, int * irp, int * ja, double * as, double * x, double * y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *m) {
		double temp = 0.0;
		for (int j = irp[i]; j < irp[i + 1]; ++j) {
			temp += as[j] * x[ja[j]];
		}
		y[i] = temp;
	}
}

__global__ void vectorMiningCSR(int * m, int * d_warp_size, int * irp, int * ja, double * as, double * x, double * y) {
	extern __shared__ volatile double sdata[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int warp_size = *d_warp_size;

	int warp = i / warp_size; 
	int lane = i & (warp_size - 1); 
	int row = warp;

	sdata[tid] = 0;

	if (row < *m) {
		for (int j = irp[row] + lane ; j < irp[row + 1]; j += warp_size)
			sdata[tid] += as[j] * x[ja[j]];

		if (warp_size == 32) { if (lane < 16) { sdata[tid] += sdata[tid + 16]; } }
		if (lane < 8) { sdata[tid] += sdata[tid + 8]; }
		if (lane < 4) { sdata[tid] += sdata[tid + 4]; }
		if (lane < 2) { sdata[tid] += sdata[tid + 2]; }
		if (lane < 1) { sdata[tid] += sdata[tid + 1]; }

		if (lane == 0)
			y[row] += sdata[tid];
	}
}

__global__ void scalarEllpack(int * m, int * ja, double * as, double * x, double * y, int * maxnz) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int rows = *m, maximum_nz = *maxnz;
	if (i < rows) {
		double temp = 0.0;
		for (unsigned int j = 0; j < *maxnz; ++j) {
			temp += as[i * maximum_nz + j] * x[ja[i * maximum_nz + j]];
		}
		y[i] = temp;
	}
}


void allocateCSR(CSR * &csr, int * &irp, int * &ja, double * &as, double * &x, double *&y, int &m, int &n) {
	int nz = csr->getnz();

	cudaMalloc((void**)&irp, sizeof(int) * (n + 1));
	cudaMalloc((void**)&ja, sizeof(int) * nz);
	cudaMalloc((void**)&as, sizeof(double) * nz);
	cudaMalloc((void**)&x, sizeof(double) * n);
	cudaMalloc((void**)&y, sizeof(double) * n);

	cudaMemcpy(irp, csr->irp, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ja, csr->getja(), sizeof(int) * nz, cudaMemcpyHostToDevice);
	cudaMemcpy(as, csr->getas(), sizeof(double) * nz, cudaMemcpyHostToDevice);
	cudaMemcpy(x, csr->getX(), sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(y, csr->y, sizeof(double) * n, cudaMemcpyHostToDevice);
}

void allocateEllpack(Ellpack * &ellpack, int * &ja, double * &as, double * &x, double * &y, int * &maxnz, int &m, int &n) {
	long long int host_maxnz = ellpack->getmaxnz();
	long long int rows = m;

	int * host_ja = ellpack->get1Dja();
	double * host_as = ellpack->get1Das();

	cudaMalloc((void**)&ja, sizeof(int) * rows * host_maxnz);
	cudaMalloc((void**)&as,  sizeof(double) * rows * host_maxnz);
	cudaMalloc((void**)&x, sizeof(double) * rows);
	cudaMalloc((void**)&y, sizeof(double) * rows);
	cudaMalloc((void**)&maxnz, sizeof(int));
	cudaMemcpy(ja, host_ja, sizeof(int) * rows * host_maxnz, cudaMemcpyHostToDevice);
	cudaMemcpy(as, host_as, sizeof(double) * rows * host_maxnz, cudaMemcpyHostToDevice);
	cudaMemcpy(x, ellpack->getX(), sizeof(double) * rows, cudaMemcpyHostToDevice);
	cudaMemcpy(y, ellpack->y, sizeof(double) * m, cudaMemcpyHostToDevice);
	cudaMemcpy(maxnz, &host_maxnz, sizeof(int), cudaMemcpyHostToDevice);
}

void deallocateCSR(int * &irp, int * &ja, double * &as, double * &x, double *&y) {
	cudaFree(irp);
	cudaFree(ja);
	cudaFree(as);
	cudaFree(x);
	cudaFree(y);
}

void deallocateEllpack(int * &ja, double * &as, double * &x, double * &y, int * &maxnz) {
	cudaFree(ja);
	cudaFree(as);
	cudaFree(x);
	cudaFree(y);
	cudaFree(maxnz);
}

void collectResults(CSR * &csr, Ellpack * &ellpack, double * &csr_y, double * &ellpack_y, int &n) {
	cudaMemcpy(csr->y, csr_y, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(ellpack->y, ellpack_y, sizeof(double) * n, cudaMemcpyDeviceToHost);
}

void getBlockNumbers(int m, int &warp_size) {
	n_blocks_scalar = m / scalar_thr_block;
	if (m % scalar_thr_block > 0.0) {
		n_blocks_scalar++;
	}
	n_blocks_vm = (m * warp_size) / vm_thr_block;
	if ((m * warp_size) % vm_thr_block > 0.0) {
		n_blocks_vm++;
	}
	if (n_blocks_vm > MAX_N_BLOCKS) {
		vm_thr_block *= 2;
		n_blocks_vm = (m * warp_size) / vm_thr_block;
		if ((m * warp_size) % vm_thr_block > 0.0) {
			n_blocks_vm++;
		}
	}
	if (n_blocks_vm > MAX_N_BLOCKS) {
		warp_size /= 2;
		n_blocks_vm = (m * warp_size) / vm_thr_block;
		if ((m * warp_size) % vm_thr_block > 0.0) {
			n_blocks_vm++;
		}
	}
}

void solveCuda(IOmanager * io, std::string path, CSR * &csr, Ellpack * &ellpack) {

	int m = csr->getRows();
	int n = csr->getCols();
	int warp_size = 32, *d_warp_size;
	
	StopWatchInterface* timer = 0;
	sdkCreateTimer(&timer);

	getBlockNumbers(m, warp_size);
	const int shmem_size = vm_thr_block * sizeof(double);
	
	int * csr_irp, * csr_ja, * ellpack_ja, * maxnz, * rows;
	double * csr_as, * csr_x, * csr_y, * ellpack_as, * ellpack_x, * ellpack_y;
	if (csr->fitsInMemory()) {
		allocateCSR(csr, csr_irp, csr_ja, csr_as, csr_x, csr_y, m, n);
	}
	if (ellpack->fitsInMemory()) {
		allocateEllpack(ellpack, ellpack_ja, ellpack_as, ellpack_x, ellpack_y, maxnz, m, n);
	}
	cudaCheckError(__LINE__);

	cudaMalloc((void**)&rows, sizeof(int));
	cudaMemcpy(rows, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_warp_size, sizeof(int));
	cudaMemcpy(d_warp_size, &warp_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError(__LINE__);

	for (int k = 0; k < NR_RUNS + 2; ++k) {
		if (csr->fitsInMemory()) {

			timer->reset();
			timer->start();
			scalarCSR<<<n_blocks_scalar, scalar_thr_block>>>(rows, csr_irp, csr_ja, csr_as, csr_x, csr_y);
			cudaDeviceSynchronize();
			timer->stop();
			csr->trackCSRTime(SCALAR, timer->getTime());
			timer->reset();

			cudaMemset(csr_y, 0.0, sizeof(double) * m);
			cudaCheckError(__LINE__);

			timer->reset();
			timer->start();
			vectorMiningCSR<<<n_blocks_vm, vm_thr_block, shmem_size>>>(rows, d_warp_size, csr_irp, csr_ja, csr_as, csr_x, csr_y);
			cudaDeviceSynchronize();
			timer->stop();
			csr->trackCSRTime(VECTOR_MINING, timer->getTime());
			timer->reset();
			cudaCheckError(__LINE__);
		}
		
		if (ellpack->fitsInMemory()) {
			timer->reset();
			timer->start();
			scalarEllpack<<<n_blocks_scalar, scalar_thr_block>>>(rows, ellpack_ja, ellpack_as, ellpack_x, ellpack_y, maxnz);
			cudaDeviceSynchronize();
			timer->stop();
			ellpack->trackTime(timer->getTime());
			timer->reset();

			cudaCheckError(__LINE__);
		}

		if (k != NR_RUNS) {
			if (csr->fitsInMemory()) cudaMemset(csr_y, 0.0, sizeof(double) * m);
			if (ellpack->fitsInMemory()) cudaMemset(ellpack_y, 0.0, sizeof(double) * m);
		}
	}

	collectResults(csr, ellpack, csr_y, ellpack_y, n);
	cudaCheckError(__LINE__);

	deallocateCSR(csr_irp, csr_ja, csr_as, csr_x, csr_y);
	deallocateEllpack(ellpack_ja, ellpack_as, ellpack_x, ellpack_y, maxnz);
	cudaCheckError(__LINE__);
	
	io->exportResults(CUDA, path, csr, ellpack);
}
