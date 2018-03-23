#ifndef UTILS_H
#define UTILS_H
#include <string>

const std::string OUTPUTS = "outputs/";

const std::string SEQUENTIAL = OUTPUTS + "sequential.csv";
const std::string OPENMP = OUTPUTS + "openmp.csv";
const std::string CUDA = OUTPUTS + "cuda.csv";

const std::string PATTERN = "matrix coordinate pattern.*";

const int NR_RUNS = 5;

#endif