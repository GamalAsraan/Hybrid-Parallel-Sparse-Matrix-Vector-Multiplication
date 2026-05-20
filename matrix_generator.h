#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <vector>

struct CSRMatrix {
    int N;
    std::vector<double> values;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};

std::vector<std::vector<double>> generateDenseSparseMatrix(int N, double sparsity = 0.95);
CSRMatrix convertToCSR(const std::vector<std::vector<double>> &mat);
void printDense(const std::vector<std::vector<double>> &mat);
void printCSR(const CSRMatrix &csr);

#endif