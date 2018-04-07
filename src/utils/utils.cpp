#include "utils.h"

double ** alloc2dDouble(int n, int m) {
    double *data = new double [m*n];
    double **array = new double *[n];
    for (int i = 0; i < n; i++) {
        array[i] = &(data[i * m]);
    }
    return array;
}

int ** alloc2d(int n, int m) {
    int *data = new int [m*n];
    int **array = new int *[n];
    for (int i = 0; i < n; i++) {
        array[i] = &(data[i * m]);
    }
    return array;
}
