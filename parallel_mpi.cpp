#include <vector>
#include <iostream>
#include "mpi_compat.h"
#include "compute_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

vector<double> spmv_csr_parallel(const CSRMatrix &mat,
                                 const vector<double> &x,
                                 int startRow,
                                 int endRow)
{
    vector<double> result(endRow - startRow, 0.0);

    #pragma omp parallel for
    for (int i = startRow; i < endRow; i++) {
        double sum = 0.0;

        for (int idx = mat.rowPtr[i]; idx < mat.rowPtr[i + 1]; idx++) {
            sum += mat.values[idx] * x[mat.colIndex[idx]];
        }

        result[i - startRow] = sum;
    }

    return result;
}