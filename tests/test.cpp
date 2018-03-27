#include <stdlib.h>
#include <string>
#include <iostream>
#include <string>
#include <regex>

#include "../src/io/mmio.h"
#include "../src/utils/utils.h"

double * readArray(std::string filename) {
	FILE * file;
	MM_typecode matcode;
	
	int M, N; 
	size_t max_nz = 0;
	size_t pointer = 0;
	double * array;
	std::string type;

	if ((file = fopen(filename.c_str(), "r")) == NULL) { 
		std::cout << "Could not open file: " << filename << std::endl;
		exit(1);
	}
	
	if (mm_read_banner(file, &matcode) != 0) {
		std::cout << "Could not process Matrix Market banner: " << filename << std::endl;
		exit(1);	
	}

	if (!mm_is_array(matcode)) {
		std::cout << "File provided for testing is not an array: " << filename << std::endl;
		exit(1);
	}

	if (mm_read_mtx_array_size(file, &M, &N) != 0) {
		std::cout << "Could not get sizes: " << filename << std::endl;
		exit(1);
	}

	array = new double[M];

	for (int i = 0; i < M; i++) {
		double value;
		if (fscanf(file, "%lg\n", &value) < 0) {
			std::cout <<  "Error reading from file of type " << type << ": " << filename << ", exiting..." << std::endl;
			exit(1);
		}
		array[i] = value;
	}
	return array;
}

bool equalSolution(double * b, double * y, int m) {
	for (int i = 0; i < m; ++i) {
		std::cout << b[i] << " vs " << y[i] << std::endl;
		if (b[i] != y[i]) {
			return false;
		}
	}
	std::cout << "true" << std::endl;
	return true;
}