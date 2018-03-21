#include "iomanager.h"
#include "mmio.h"
#include "element.h"
#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "iomanager.h"

#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <map>
#include <set>
#include <string>
#include <regex>
IOmanager::IOmanager() {}

std::string IOmanager::parseArguments(int argc, char ** argv) {
	if (argc != 2) {
		std::cout << "Wrong usage of arguments, one should use: solver <path>" << std::endl;
		exit(1);
	}
	return string(argv[1]);
}

std::pair<CSR*, Ellpack*> IOmanager::readFile(string filename) {
	FILE * file;
	MM_typecode matcode;
	
	CSR * csr; 
	Ellpack * ellpack;
	
	int M, N, nz; 
	map<int, set<Element*>> occurences;
	size_t max_nz = 0;
	size_t pointer = 0;

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

		if (fscanf(file, "%d %d %lg\n", &m, &n, &value) < 0) {
			std::cout <<  "Error reading from file " << filename << ", exiting..." << std::endl;
            exit(1);
		}

		m--; n--;
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
	
	csr = new CSR(N, M, nz);
	ellpack = new Ellpack(N, M, max_nz);	
	
	for (auto map_it = occurences.begin(); map_it != occurences.end(); map_it++) {
		int col_index = map_it->first;
		set<Element *> elements = map_it->second;
		csr->addPointer(pointer);
		for (auto set_it = elements.begin(); set_it != elements.end(); set_it++) {
			double value = (*set_it)->getValue();
			int row_index = (*set_it)->getRow();
			csr->addElement(col_index, value);
			ellpack->addElement(col_index, row_index, value);
			pointer++;
		}
	}
	
	return make_pair(csr, ellpack);
}

std::string IOmanager::extractName(std::string path) {
	std::regex reg("[A-Za-z_0-9]+.mtx");
	std::smatch match;
	if (std::regex_search(path, match, reg) && match.size() == 1) {
		std::string name = match[0]; 
		return name.substr(0, name.length() - 4);
	} else {
		std::cout << "Couldn't find a proper matrix name for " << path << std::endl;
		return path;
	}
}

void IOmanager::exportResults(std::string output_file, std::string path, CSR * csr, Ellpack * ellpack) {
	std::ofstream out; std::string name;
	out.open(output_file, std::ios::out | std::ios_base::app);

	if (out.fail()) {
		std::cout << "Could not open Sequential output file to export results" << std::endl; 
		return;
	}

	csr->printElapsedTime();
	ellpack->printElapsedTime();
	
	name = extractName(path);
	out << name << ", " << csr->getMegaFlops() << ", " << ellpack->getMegaFlops() << "," << std::endl;
}

