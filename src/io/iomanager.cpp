#include "iomanager.h"
#include "mmio.h"
#include "element.h"
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <unordered_map>
#include <set>

IOmanager::IOmanager() {}

Matrix * IOmanager::readFile(string filename) {
	FILE * file;
	MM_typecode matcode;
	int M, N, nz; 
	unordered_map<int, set<Element*>> occurences;
	size_t max_nz = 0;

	if ((file = fopen(filename.c_str(), "r")) == NULL) { 
		std::cout << "Could not open file: " << filename << std::endl;
		exit(1);
	}
	
	if (mm_read_banner(file, &matcode) != 0) {
		std::cout << "Could not process Matrix Market banner: " << filename << std::endl;
	       	exit(1);	
	}

	if (!mm_is_sparse(matcode)) {
		std::cout << "This application supports sparse matrices only: " << filename << std::endl;
		exit(1);
	}

	if (mm_read_mtx_crd_size(file, &M, &N, &nz) != 0) {
		std::cout << "Could not get sizes: " << filename << std::endl;
		exit(1);
	}

	for (int i = 0; i < nz; i++) {
		int m, n;
		double value;

		fscanf(file, "%d %d %lg\n", &m, &n, &value);
		Element * el = new Element(m, value);
		
		auto it = occurences.find(n);
		if (it != occurences.end()) {
			it->second.insert(el);
			if (max_nz < it->second.size()) {
				max_nz = it->second.size();
			}
		} else {
			occurences.insert({n, {el}});
		}
	}

	Matrix * m = new Matrix(N, M);	
	return m;
}
