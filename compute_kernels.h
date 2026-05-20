#ifndef COMPUTE_KERNELS_H
#define COMPUTE_KERNELS_H

#include <vector>

#include "matrix_generator.h"

std::vector<double> multiplyDense(const std::vector<std::vector<double>> &mat,
                                  const std::vector<double> &x);
std::vector<double> multiplyCSR(const CSRMatrix &mat,
                                const std::vector<double> &x);
std::vector<double> spmv_csr_parallel(const CSRMatrix &mat,
                                      const std::vector<double> &x,
                                      int startRow,
                                      int endRow);

#endif