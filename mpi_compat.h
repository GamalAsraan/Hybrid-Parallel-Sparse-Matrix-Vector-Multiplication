#ifndef MPI_COMPAT_H
#define MPI_COMPAT_H

#if defined(__has_include) && __has_include(<mpi.h>)
#include <mpi.h>
#define PND_HAS_MPI 1
#else
#define PND_HAS_MPI 0

using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
constexpr int MPI_INT = 0;
constexpr int MPI_DOUBLE = 0;

inline int MPI_Init(int *, char ***) { return 0; }
inline int MPI_Finalize() { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int *rank)
{
    *rank = 0;
    return 0;
}
inline int MPI_Comm_size(MPI_Comm, int *size)
{
    *size = 1;
    return 0;
}
inline int MPI_Bcast(void *, int, int, int, MPI_Comm) { return 0; }
inline int MPI_Barrier(MPI_Comm) { return 0; }
inline int MPI_Gatherv(const void *, int, int, void *, const int *, const int *, int, int, MPI_Comm)
{
    return 0;
}
#endif

#endif