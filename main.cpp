#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "compute_kernels.h"
#include "mpi_compat.h"

using namespace std;

static void broadcastCSRMatrix(CSRMatrix &mat, int rank)
{
    int nnz = (rank == 0) ? static_cast<int>(mat.values.size()) : 0;

    MPI_Bcast(&mat.N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        mat.values.resize(nnz);
        mat.colIndex.resize(nnz);
        mat.rowPtr.resize(mat.N + 1);
    }

    MPI_Bcast(mat.rowPtr.data(), static_cast<int>(mat.rowPtr.size()), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat.values.data(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat.colIndex.data(), nnz, MPI_INT, 0, MPI_COMM_WORLD);
}

static void computeRowBounds(int N, int rank, int size, int &startRow, int &endRow)
{
    int baseRows = N / size;
    int remainder = N % size;

    startRow = rank * baseRows + min(rank, remainder);
    int localRows = baseRows + (rank < remainder ? 1 : 0);
    endRow = startRow + localRows;
}

static double secondsSince(const chrono::high_resolution_clock::time_point &start,
                           const chrono::high_resolution_clock::time_point &end)
{
    return chrono::duration<double>(end - start).count();
}

static void writeCsv(const string &path,
                     const vector<int> &ns,
                     const vector<double> &denseTimes,
                     const vector<double> &csrTimes,
                     const vector<double> &mpiComputeTimes,
                     const vector<double> &mpiTotalTimes)
{
    ofstream out(path);
    out << "N,dense_sec,csr_sec,mpi_compute_sec,mpi_total_sec\n";
    for (size_t i = 0; i < ns.size(); ++i) {
        out << ns[i] << "," << denseTimes[i] << "," << csrTimes[i]
            << "," << mpiComputeTimes[i] << "," << mpiTotalTimes[i] << "\n";
    }
}

static void writeSpeedupCsv(const string &path,
                            const vector<int> &ns,
                            const vector<double> &csrTimes,
                            const vector<double> &mpiTotalTimes,
                            int mpiSize)
{
    ofstream out(path);
    out << "N,speedup_total,efficiency_total\n";
    for (size_t i = 0; i < ns.size(); ++i) {
        double speedup = (mpiTotalTimes[i] > 0.0) ? (csrTimes[i] / mpiTotalTimes[i]) : 0.0;
        double efficiency = (mpiSize > 0) ? (speedup / mpiSize) : 0.0;
        out << ns[i] << "," << speedup << "," << efficiency << "\n";
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double sparsity = 0.95;
    vector<int> ns = {1000, 2000, 3000, 4000, 5000, 6000};
    int warmupRuns = 1;
    int measureRuns = 5;
    vector<double> denseTimes;
    vector<double> csrTimes;
    vector<double> mpiComputeTimes;
    vector<double> mpiTotalTimes;

    denseTimes.reserve(ns.size());
    csrTimes.reserve(ns.size());
    mpiComputeTimes.reserve(ns.size());
    mpiTotalTimes.reserve(ns.size());

    for (int N : ns) {
        CSRMatrix mat;
        vector<vector<double>> dense;
        vector<double> x(N, 1.0);

        if (rank == 0) {
            dense = generateDenseSparseMatrix(N, sparsity);
            mat = convertToCSR(dense);
        }

        broadcastCSRMatrix(mat, rank);
        MPI_Bcast(x.data(), static_cast<int>(x.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int w = 0; w < warmupRuns; ++w) {
                multiplyDense(dense, x);
                multiplyCSR(mat, x);
            }

            double denseSum = 0.0;
            double csrSum = 0.0;

            for (int r = 0; r < measureRuns; ++r) {
                auto denseStart = chrono::high_resolution_clock::now();
                auto denseResult = multiplyDense(dense, x);
                auto denseEnd = chrono::high_resolution_clock::now();
                denseSum += secondsSince(denseStart, denseEnd);

                auto csrStart = chrono::high_resolution_clock::now();
                auto csrResult = multiplyCSR(mat, x);
                auto csrEnd = chrono::high_resolution_clock::now();
                csrSum += secondsSince(csrStart, csrEnd);
            }

            denseTimes.push_back(denseSum / measureRuns);
            csrTimes.push_back(csrSum / measureRuns);
        }

        int startRow = 0;
        int endRow = 0;
        computeRowBounds(mat.N, rank, size, startRow, endRow);

        vector<double> finalResult;
        vector<int> recvCounts;
        vector<int> displs;

        if (rank == 0) {
            finalResult.resize(mat.N);
            recvCounts.resize(size);
            displs.resize(size);

            int offset = 0;
            for (int proc = 0; proc < size; ++proc) {
                int procStart = 0;
                int procEnd = 0;
                computeRowBounds(mat.N, proc, size, procStart, procEnd);
                recvCounts[proc] = procEnd - procStart;
                displs[proc] = offset;
                offset += recvCounts[proc];
            }
        }

        for (int w = 0; w < warmupRuns; ++w) {
            MPI_Barrier(MPI_COMM_WORLD);
            vector<double> localWarm = spmv_csr_parallel(mat, x, startRow, endRow);
            MPI_Gatherv(localWarm.data(), static_cast<int>(localWarm.size()), MPI_DOUBLE,
                        finalResult.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        double mpiComputeSum = 0.0;
        double mpiTotalSum = 0.0;

        for (int r = 0; r < measureRuns; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            auto totalStart = chrono::high_resolution_clock::now();

            auto computeStart = chrono::high_resolution_clock::now();
            vector<double> local = spmv_csr_parallel(mat, x, startRow, endRow);
            auto computeEnd = chrono::high_resolution_clock::now();

            MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                        finalResult.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            auto totalEnd = chrono::high_resolution_clock::now();

            if (rank == 0) {
                mpiComputeSum += secondsSince(computeStart, computeEnd);
                mpiTotalSum += secondsSince(totalStart, totalEnd);
            }
        }

        if (rank == 0) {
            mpiComputeTimes.push_back(mpiComputeSum / measureRuns);
            mpiTotalTimes.push_back(mpiTotalSum / measureRuns);
            cout << "N=" << N << " done" << endl;
        }
    }

    if (rank == 0) {
        writeCsv("graphs/times.csv", ns, denseTimes, csrTimes, mpiComputeTimes, mpiTotalTimes);
        writeSpeedupCsv("graphs/speedup.csv", ns, csrTimes, mpiTotalTimes, size);
        cout << "Wrote graphs/times.csv" << endl;
        cout << "Wrote graphs/speedup.csv" << endl;
    }

    MPI_Finalize();
    return 0;
}