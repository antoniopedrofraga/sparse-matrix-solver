#!/bin/bash
cat outputs/sequential.csv | sort -V > outputs/sequential_sorted.csv
cat outputs/openmp.csv | sort -V > outputs/openmp_sorted.csv
cat outputs/cuda.csv | sort -V > outputs/cuda_sorted.csv