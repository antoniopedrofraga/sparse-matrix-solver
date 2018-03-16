#include <iostream>
#include <string>

#include "matrix/csr.h"
#include "matrix/ellpack.h"
#include "io/iomanager.h"
#include "matrix/matrix.h"

void solveCuda(IOmanager * io, std::string path, CSR * &csr, Ellpack * &ellpack);

int main(int argc, char ** argv) {	
	IOmanager * io = new IOmanager();
	std::string path = io->parseArguments(argc, argv);
	std::pair<CSR*, Ellpack*> matrices = io->readFile(path);

	solveCuda(io, path, matrices.first, matrices.second);
}