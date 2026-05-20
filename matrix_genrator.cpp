#include <iostream>
#include <vector>
#include <random>

using namespace std;

#include "matrix_generator.h"

vector<vector<double>> generateDenseSparseMatrix(int N, double sparsity) {
    vector<vector<double>> mat(N, vector<double>(N, 0.0));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> valDist(1.0, 10.0);
    uniform_real_distribution<double> prob(0.0, 1.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            if (prob(gen) > sparsity) {
                mat[i][j] = valDist(gen);
            }
        }
    }

    return mat;
}

CSRMatrix convertToCSR(const vector<vector<double>> &mat) {
    CSRMatrix csr;
    int N = mat.size();
    csr.N = N;
    csr.rowPtr.push_back(0);

    int nnz = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            if (mat[i][j] != 0.0) {
                csr.values.push_back(mat[i][j]);
                csr.colIndex.push_back(j);
                nnz++;
            }
        }
        csr.rowPtr.push_back(nnz);
    }

    return csr;
}

void printDense(const vector<vector<double>> &mat) {
    for (auto &row : mat) {
        for (auto v : row) {
            cout << v << " ";
        }
        cout << endl;
    }
}

void printCSR(const CSRMatrix &csr) {
    cout << "Values: ";
    for (auto v : csr.values) cout << v << " ";
    cout << "\nColIndex: ";
    for (auto c : csr.colIndex) cout << c << " ";
    cout << "\nRowPtr: ";
    for (auto r : csr.rowPtr) cout << r << " ";
    cout << endl;
}