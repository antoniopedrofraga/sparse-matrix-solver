#include <stdlib.h>
#include <string>
#include <iostream>
#include <string>
#include <regex>
#include <math.h>

#include "../src/io/mmio.h"
#include "../src/utils/utils.h"

double * readArray(std::string filename) {
	FILE * file;
	MM_typecode matcode;
	
	int M, N; 
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
		if (fabs(b[i] - y[i]) > 1.e-06) {
			std::cout << "At i = " << i << " -> " << b[i] << " vs " << y[i] << std::endl;
			std::cout << "False" << std::endl;
			return false;
		}
	}
	return true;
}