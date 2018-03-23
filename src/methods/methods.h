#ifndef METHODS_H
#define METHODS_H

#include "../io/iomanager.h"
#include "../matrix/matrix.h"
#include "../matrix/csr.h"
#include "../matrix/ellpack.h"
#include "../utils/utils.h"

void sequentialCSR(CSR * &csr);
void sequentialEllpack(Ellpack * &ellpack);

void openmpCSR(CSR * &csr);
void openmpEllpack(Ellpack * &ellpack);

void solveCuda(IOmanager * io, std::string path, CSR * &csr, Ellpack * &ellpack);

#endif