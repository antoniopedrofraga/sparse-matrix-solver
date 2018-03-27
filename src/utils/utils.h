#ifndef UTILS_H
#define UTILS_H
#include <string>

const std::string OUTPUTS = "outputs/";

const std::string SEQUENTIAL = OUTPUTS + "sequential.csv";
const std::string OPENMP = OUTPUTS + "openmp.csv";
const std::string CUDA = OUTPUTS + "cuda.csv";

const std::string PATTERN = "matrix coordinate pattern.*";
const std::string ARRAY = "matrix coordinate pattern.*";

const int NR_RUNS = 10;

const int MIN_THREADS = 2;
const int MAX_THREADS = 16;
const int NUM_THREADS = MAX_THREADS - MIN_THREADS + 1;

#endif