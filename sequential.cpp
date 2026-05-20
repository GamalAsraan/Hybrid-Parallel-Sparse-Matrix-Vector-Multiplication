#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

#include "compute_kernels.h"

vector<double> multiplyDense(const vector<vector<double>> &mat, const vector<double> &x) {
    int N = mat.size();
    vector<double> result(N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i] += mat[i][j] * x[j];
        }
    }

    return result;
}

vector<double> multiplyCSR(const CSRMatrix &mat, const vector<double> &x) {
    int N = mat.N;
    vector<double> result(N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int idx = mat.rowPtr[i]; idx < mat.rowPtr[i + 1]; idx++) {
            result[i] += mat.values[idx] * x[mat.colIndex[idx]];
        }
    }

    return result;
}

