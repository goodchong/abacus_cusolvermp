#pragma once
#include "mpi.h"

#include <complex>
#include <fstream>
#include <vector>
#include <cal.h>
#include <cusolverMp.h>
#include "module_base/macros.h"

template<typename inputT>
class Diag_CusolverMP_gvd
{
  private:
    using outputT = typename GetTypeReal<inputT>::type;

  public:
    Diag_CusolverMP_gvd(const MPI_Comm comm,
                          const int nev,
                          const int narows,
                          const int nacols,
                          const int* desc);

    int generalized_eigenvector(inputT* A,
                                inputT* B,
                                outputT* EigenValue,
                                inputT* EigenVector);
    ~Diag_CusolverMP_gvd();
    void outputParameters();
  private:
    MPI_Comm comm;
    int nFull;
    int nev;
    int narows;
    int nacols;
    int cblacs_ctxt;
    int nblk;
    int lda;
    int nprows;
    int npcols;
    int myprow;
    int mypcol;
    int comm_f;
    int my_global_mpi_id;
    int mpi_size;
    cudaDataType_t datatype;

    cal_comm_t cal_comm = NULL;
    cudaStream_t localStream = NULL;
    cusolverMpHandle_t cusolverMpHandle = NULL;
    cusolverMpGrid_t grid = NULL;

    /* cusolverMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverMpMatrixDescriptor_t descB = NULL;
    cusolverMpMatrixDescriptor_t descZ = NULL;

    // TODO 这个参数设法应该不对
    int64_t matrix_i;
    int64_t matrix_j;

    void *d_A = NULL;
    void *d_B = NULL;
    void *d_D = NULL;
    void *d_Z = NULL;

};

// 实现模板类的成员函数

