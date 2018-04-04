#include <stdio.h>
#include <iostream>

#include "../matrix/matrix.h"
#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../io/iomanager.h"
#include "../utils/utils.h"


const int BLOCKS_PER_MP = 32; // Sufficiently large for memory transaction hiding
int thread_block = 1024; // Must be a power of 2 >= 64
int max_blocks = 0;   // Blocks in a grid
int red_sz = 0;       // Size of reduction buffer
double *o_data = NULL, *d_res_data = NULL, *h_res_data = NULL;
static struct cudaDeviceProp *prop = NULL;


void cudaCheckError() {
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
		exit(0);
	}
}

float cpuReduction(int n, double * x) {
	float result = 0.0f;
	for (int i = 0; i < n; ++i) {
		result += x[i];
	}
	return result;
}

__device__ void warpReduce(double *sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid +  8];
	sdata[tid] += sdata[tid +  4];
	sdata[tid] += sdata[tid +  2];
	sdata[tid] += sdata[tid +  1];
}

void reduce_alloc_wrk() {
	if (prop == NULL) {
		if ((prop = (struct cudaDeviceProp *) malloc(sizeof(struct cudaDeviceProp))) == NULL) {
			fprintf(stderr,"CUDA Error gpuInit3: not malloced prop\n");
			return;
		}
		cudaSetDevice(0); // BEWARE: you may have more than one device
		cudaGetDeviceProperties(prop, 0); 
	}
	if (thread_block <= 0)
		std::cerr << "thread_block must be a power of 2 between 64 and 1024" << std::endl;

	if (max_blocks == 0) {
		int mpCnt;
		mpCnt = prop->multiProcessorCount;
		max_blocks = mpCnt * BLOCKS_PER_MP;
		red_sz = (max_blocks + thread_block - 1) / thread_block;
	}
	
	if (o_data == NULL) cudaMalloc(&o_data, max_blocks * sizeof(double));
	if (d_res_data == NULL) cudaMalloc(&d_res_data, (red_sz) * sizeof(double));
	if (h_res_data == NULL) h_res_data = (double *)malloc((red_sz) * sizeof(double));
}


template <unsigned int THD> __global__ void reduceCSR(int n, double *g_idata, double *g_odata) {
	extern __shared__ double sdata[];
		// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int gridSize = blockDim.x * gridDim.x;
	double temp = 0.0;

	for (int j = csr->irp[i]; j < csr->irp[i + 1] - 1; ++j) {
		temp += csr->as[j] * csr->x[csr->ja[j]];
	}
	g_odata[i] = temp;

	__syncthreads();
		// do reduction in shared mem
	if (THD >= 1024){ if (tid < 512) { sdata[tid] += sdata[tid + 512]; }  __syncthreads();  }
	if (THD >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; }  __syncthreads();  }
	if (THD >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; }  __syncthreads();  }
	if (THD >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid +  64]; }  __syncthreads();  }
	
	// write result for this block to global mem
	//if (tid < 32) warpReduce(sdata,tid);
	if (tid == 0) g_odata[blockIdx.x] += sdata[0];
}

void do_gpu_reduce(int m, CSR * c_csr, double *g_odata) {
	const int shmem_size = thread_block * sizeof(double);
	int nblocks = (m + thread_block - 1) / thread_block;
	if (nblocks > max_blocks) nblocks = max_blocks;

	switch(thread_block) {
		case 1024:
		reduceCSR<1024><<<nblocks, 1024, shmem_size,0>>>(m, c_csr, g_odata);
		break;
		case 512:
		reduceCSR<512><<<nblocks, 512, shmem_size,0>>>(m, c_csr, g_odata); 
		break;
		case 256:
		reduceCSR<256><<<nblocks, 256, shmem_size,0>>>(m, c_csr, g_odata); 
		break;
		case 128:
		reduceCSR<128><<<nblocks, 128, shmem_size,0>>>(m, c_csr, g_odata); 
		break;
		case 64:
		reduceCSR<64><<<nblocks, 64, shmem_size,0>>>(m, c_csr, g_odata); 
		break;
		default:
		std::cerr << "thread_block must be a power of 2 between 64 and 1024" << std::endl;
	}
	return;  
}

double gpu_reduce(int m, double *d_v) {
	reduce_alloc_wrk();
	cudaMemset((void *)o_data, 0, max_blocks * sizeof(double));
	cudaMemset((void *)d_res_data, 0, red_sz * sizeof(double));

	do_gpu_reduce(m, d_v, o_data);
	do_gpu_reduce(max_blocks, o_data, d_res_data);
	cudaError_t err = cudaMemcpy(h_res_data, d_res_data, red_sz * sizeof(double), cudaMemcpyDeviceToHost);
	return(cpuReduction(red_sz, h_res_data));
}


__global__ void solveCSR(CSR * csr) {
	int i = threadIdx.x;
	double temp = 0.0;
	for (int j = csr->irp[i]; j < csr->irp[i + 1] - 1; ++j) {
		temp += csr->as[j] * csr->x[csr->ja[j]];
	}
	csr->y[i] = temp;
}

__global__ void solveEllpack(Ellpack * ellpack) {
	int i = threadIdx.x;
	double temp = 0.0;
	for (int j = 0; j < ellpack->maxnz; ++j) {
		temp += ellpack->as[i][j] * ellpack->x[ellpack->ja[i][j]];
	}
	ellpack->y[i] = temp;
}

void solveCuda(IOmanager * io, std::string path, CSR * &csr, Ellpack * &ellpack) {
	
	const int m = csr->getRows();
	const int csize = sizeof(CSR);
	const int esize = sizeof(Ellpack);
	
	CSR * csr_c;
	Ellpack * ellpack_c;

	reduce_alloc_wrk();

	const int shmem_size = thread_block * sizeof(double);
	int nblocks = (m + thread_block - 1) / thread_block;
	if (nblocks > max_blocks) nblocks = max_blocks;

	cudaMalloc((void**)&csr_c, csize);
	cudaMalloc((void**)&ellpack_c, esize);
	cudaMemcpy(csr_c, csr, csize, cudaMemcpyHostToDevice); 
	cudaMemcpy(ellpack_c, ellpack, esize, cudaMemcpyHostToDevice);

	
	for (int k = 0; k < NR_RUNS; ++k) {
		csr->trackTime();
		solveCSR<<<1, m>>>(csr_c);
		cudaCheckError();
		csr->trackTime();
		
		ellpack->trackTime();
		solveEllpack<<<1, m>>>(ellpack_c);
		cudaCheckError();
		ellpack->trackTime();
	}
	
	io->exportResults(CUDA, path, csr, ellpack);

	cudaMemcpy(csr, csr_c, csize, cudaMemcpyDeviceToHost); 
	cudaMemcpy(ellpack, ellpack_c, esize, cudaMemcpyDeviceToHost);

	cudaFree(csr_c);
	cudaFree(ellpack_c);
}
