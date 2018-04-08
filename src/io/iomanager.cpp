#include "iomanager.h"
#include "mmio.h"
#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "iomanager.h"

#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <map>
#include <vector>
#include <string>
#include <regex>
#include <math.h>

IOmanager::IOmanager() {}

std::pair<CSR*, Ellpack*> IOmanager::readFile(string filename) {
	FILE * file;
	MM_typecode matcode;
	
	CSR * csr; 
	Ellpack * ellpack;
	
	int M, N, num_values, nz; 
	map<int, std::vector<std::pair<int, double>>> occurences;
	size_t max_nz = 1;
	size_t pointer = 0;
	std::string type;

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

	if (mm_read_mtx_crd_size(file, &M, &N, &num_values) != 0) {
		std::cout << "Could not get sizes: " << filename << std::endl;
		exit(1);
	}

	nz = num_values;
	bool is_symmetric = mm_is_symmetric(matcode);
	bool is_pattern = mm_is_pattern(matcode);

	std::cout << (is_pattern ? "(pattern) " : "(not pattern) ");
	std::cout << (is_symmetric ? "(symmetric) " : "(not symmetric) ");

	for (int i = 0; i < num_values; ++i) {
		int m, n;
		double value = 1.0;
		if (is_pattern) {
			if (fscanf(file, "%d %d\n", &m, &n) < 0) {
				std::cout <<  "Error reading from file of type " << type << ": " << filename << ", exiting..." << std::endl;
				exit(1);
			}
		} else {
			if (fscanf(file, "%d %d %lg\n", &m, &n, &value) < 0) {
				std::cout <<  "Error reading from file of type " << type << ": " << filename << ", exiting..." << std::endl;
				exit(1);
			}
		}

		m--; n--;

		auto it = occurences.find(m);
		if (it != occurences.end()) {
			it->second.push_back({n, value});
			if (max_nz < it->second.size()) {
				max_nz = it->second.size();
			}
		} else {
			occurences.insert({m, { {n, value} }});
		}

		/*
			If symmetric, add an element with row and col swapped.
		*/
		if (is_symmetric && n != m) {
			++nz;
			auto it_b = occurences.find(n);
			if (it_b != occurences.end()) {
				it_b->second.push_back({m, value});
				if (max_nz < it_b->second.size()) {
					max_nz = it_b->second.size();
				}
			} else {
				occurences.insert({n, { {m, value} }});
			}
		}
	}

	std::cout << "(M = " << M << ", N = " << N << ", MAXNZ = " << max_nz << ", NZ = " << nz << ") " << std::endl;

	csr = new CSR(N, M, nz);
	ellpack = new Ellpack(N, M, max_nz, nz);

	for (auto map_it = occurences.begin(); map_it != occurences.end(); map_it++) {
		int row_index = map_it->first;
		std::vector<std::pair<int, double>> elements = map_it->second;
		csr->mean += elements.size();
		ellpack->mean += elements.size();
	}

	csr->mean /= occurences.size();
	ellpack->mean /= occurences.size();

	for (auto map_it = occurences.begin(); map_it != occurences.end(); map_it++) {
		int row_index = map_it->first;
		std::vector<std::pair<int, double>> elements = map_it->second;
		if (csr->fitsInMemory()) csr->addPointer(pointer);

		csr->average_deviation += fabs(csr->mean - elements.size());
		ellpack->average_deviation += fabs(ellpack->mean - elements.size());
		for (size_t i = 0; i < elements.size(); ++i) {
			double value = elements[i].second;
			int col_index = elements[i].first;
			if (csr->fitsInMemory()) csr->addElement(col_index, value);

			if (ellpack->fitsInMemory()) {
				ellpack->addElement(row_index, col_index, value);
			}

			pointer++;
		}
	}
	if (csr->fitsInMemory()) csr->addPointer(pointer);
	csr->average_deviation /= occurences.size();
	csr->average_deviation = csr->average_deviation / csr->mean * 100;
	ellpack->average_deviation /= occurences.size();
	ellpack->average_deviation = ellpack->average_deviation / ellpack->mean * 100;

	std::cout << "(Average nz deviation: " << csr->average_deviation << "%) ";
	std::cout << (!csr->fitsInMemory() ? "(CSR doesn't fit in mem)" : "");
	std::cout << (!ellpack->fitsInMemory() ? "(Ellpack doesn't fit in mem)" : "");

	return make_pair(csr, ellpack);
}

std::string IOmanager::extractName(std::string path) {
	std::regex reg("[A-Za-z_0-9]+.mtx");
	std::smatch match;
	if (std::regex_search(path, match, reg) && match.size() == 1) {
		std::string name = match[0]; 
		return name.substr(0, name.length() - 4);
	} else {
		std::cout << "Couldn't find a propper matrix name for " << path << std::endl;
		return path;
	}
}

void IOmanager::exportResults(std::string output_file, std::string path, CSR * csr, Ellpack * ellpack) {
	std::ofstream out;
	out.open(output_file, std::ios::out | std::ios_base::app);

	if (out.fail()) {
		std::cout << "Could not open Sequential output file to export results" << std::endl; 
		return;
	}

	if (output_file != CUDA) {
		csr->printElapsedTime();
		out << csr->getnz() << " " << csr->getMegaFlops() << " " << ellpack->getMegaFlops() << std::endl;
	} else {
		csr->printElapsedCUDATime();
		out << csr->getnz() << " " << csr->getScalarMegaFlops() << " " << csr->getVecMinMegaFlops() << " " << ellpack->getMegaFlops() << std::endl;
	}
	ellpack->printElapsedTime();

	if (output_file == OPENMP) {
		this->exportOMPResults(path, csr, ellpack);
	}

	out.close();
}


void IOmanager::exportOMPResults(std::string path, CSR * csr, Ellpack * ellpack) {
	std::ofstream out; 
	std::string name = extractName(path), output_file = OUTPUTS + "/OMP_" + name + ".csv";
	out.open(output_file, std::ios::out | std::ios_base::app);

	for (int i = 0; i < NUM_THREADS; ++i) {
		out << (i + MIN_THREADS) << " " << csr->getMegaFlops(i) << " " << ellpack->getMegaFlops(i) << std::endl;
	}

	out.close();
}

