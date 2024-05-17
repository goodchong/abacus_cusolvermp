#include <assert.h>
#include "helper_cuda.h"
#include "diag_cusolvermp.cuh"
#include "module_base/global_variable.h"

extern "C"
{
#include "module_hsolver/genelpa/Cblacs.h"
}
#include "iostream"
void check_cusolver(cusolverStatus_t result, char const *const func, const char *const file,
           int const line) {
  if (result != CUSOLVER_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}
// the error check is use for cusolverMP and cuSolver
#define checkCusolverErrors(val) check_cusolver((val), #val, __FILE__, __LINE__)

static calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request)
{
    MPI_Request req;
    int         err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, (MPI_Comm)(data), &req);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    *request = (void*)(req);
    return CAL_OK;
}

static calError_t request_test(void* request)
{
    MPI_Request req = (MPI_Request)(request);
    int         completed;
    int         err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

static calError_t request_free(void* request)
{
    return CAL_OK;
}

template <typename inputT>
Diag_CusolverMP_gvd<inputT>::Diag_CusolverMP_gvd(const MPI_Comm comm,
                                                    const int nev,
                                                    const int narows,
                                                    const int nacols,
                                                    const int *desc)
{
    // 构造函数的实现
    this->comm = comm;
    this->nev = nev;
    this->narows = narows;
    this->nacols = nacols;

    this->cblacs_ctxt = desc[1];
    this->nFull = desc[2];
    this->nblk = desc[4];
    this->lda = desc[8];

    MPI_Comm_size(comm, &this->mpi_size);
    MPI_Comm_rank(comm, &(this->my_global_mpi_id));
    int localRank;
    {
        MPI_Comm localComm;

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        // warning: this is not a good way to assign devices, user should assign One process per GPU
        checkCudaErrors(cudaSetDevice(localRank % deviceCount));
    }

    Cblacs_gridinfo(this->cblacs_ctxt, &this->nprows, &this->npcols, &this->myprow, &this->mypcol);

    this->cal_comm = NULL;
    cal_comm_create_params_t params;
    params.allgather = allgather;
    params.req_test = request_test;
    params.req_free = request_free;
    params.data = (void *)(MPI_COMM_WORLD);
    params.rank = this->my_global_mpi_id;
    params.nranks = this->mpi_size;
    params.local_device = localRank;

    int calStat = calStat = cal_comm_create(params, &this->cal_comm);
    assert(calStat == CAL_OK);

    checkCudaErrors(cudaStreamCreate(&this->localStream));
    checkCusolverErrors(cusolverMpCreate(&cusolverMpHandle, localRank, this->localStream));

    
    // TODO 这个怎么处理还没想好。我们有了blacs的grid，现在又有GPU的device grid， 这两个grid是不是可以合并？我们应该怎么抽象这个模型？
    const int numRowDevices = this->nprows;
    const int numColDevices = this->npcols;

    // TODO 这个参数设法应该不对
    this->matrix_i = 1;
    this->matrix_j = 1;

    if (std::is_same<inputT, std::complex<double>>::value) 
    {
        this->datatype = CUDA_C_64F;
    } else if (std::is_same<inputT, double>::value)
    {
        this->datatype = CUDA_R_64F;
    } else 
    {
        GlobalV::ofs_running << "error cusolvermp input type" << std::endl;
    }

    checkCusolverErrors(cusolverMpCreateDeviceGrid(cusolverMpHandle, &this->grid, cal_comm, numRowDevices, numColDevices, CUSOLVERMP_GRID_MAPPING_COL_MAJOR));

    checkCudaErrors(cudaMalloc((void **)&this->d_A, this->narows * this->nacols * sizeof(inputT)));
    checkCudaErrors(cudaMalloc((void **)&this->d_B, this->narows * this->nacols * sizeof(inputT)));
    checkCudaErrors(cudaMalloc((void **)&this->d_D, this->nFull * sizeof(outputT)));
    checkCudaErrors(cudaMalloc((void **)&this->d_Z, this->narows * this->nacols * sizeof(inputT)));

    cusolverMpCreateMatrixDesc(&descA, this->grid, this->datatype, nFull, nFull, nblk, nblk, 0, 0, this->lda);
    cusolverMpCreateMatrixDesc(&descB, this->grid, this->datatype, nFull, nFull, nblk, nblk, 0, 0, this->lda);
    cusolverMpCreateMatrixDesc(&descZ, this->grid, this->datatype, nFull, nFull, nblk, nblk, 0, 0, this->lda);
    outputParameters();
    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename inputT>
Diag_CusolverMP_gvd<inputT>::~Diag_CusolverMP_gvd()
{
    
    checkCudaErrors(cudaFree(this->d_A));
    checkCudaErrors(cudaFree(this->d_B));
    checkCudaErrors(cudaFree(this->d_D));
    checkCudaErrors(cudaFree(this->d_Z));

    
    cusolverMpDestroyMatrixDesc(this->descA);
    cusolverMpDestroyMatrixDesc(this->descB);
    cusolverMpDestroyMatrixDesc(this->descZ);

    cusolverMpDestroyGrid(this->grid);

    cusolverMpDestroy(this->cusolverMpHandle);
    int calStat = cal_comm_barrier(this->cal_comm, this->localStream);
    assert(calStat == CAL_OK);
    calStat = cal_comm_destroy(this->cal_comm);
    assert(calStat == CAL_OK);
    checkCudaErrors(cudaStreamDestroy(this->localStream));
    // TODO temp solution
    checkCudaErrors(cudaSetDevice(0));
}


template<typename inputT>
int Diag_CusolverMP_gvd<inputT>::generalized_eigenvector(inputT* A,
                          inputT* B,
                          outputT* EigenValue,
                          inputT* EigenVector)
{

    GlobalV::ofs_running << "run cusolvermp!!" << std::endl;

    checkCudaErrors(cudaMemcpy(this->d_A, (void*)A, this->narows * this->nacols * sizeof(inputT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->d_B, (void*)B, this->narows * this->nacols * sizeof(inputT), cudaMemcpyHostToDevice));
    int calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);
    GlobalV::ofs_running << "run cusolvermp 2!!" << std::endl;

    /* size of workspace on device */
    size_t sygvdWorkspaceInBytesOnDevice = 0;

    /* size of workspace on host */
    size_t sygvdWorkspaceInBytesOnHost = 0;

    cusolverMpSygvd_bufferSize(cusolverMpHandle,
                                CUSOLVER_EIG_TYPE_1,
                                CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_LOWER,
                                this->nFull,
                                this->matrix_i,
                                this->matrix_j,
                                descA,
                                this->matrix_i,
                                this->matrix_j,
                                descB,
                                this->matrix_i,
                                this->matrix_j,
                                descZ,
                                this->datatype,
                                &sygvdWorkspaceInBytesOnDevice,
                                &sygvdWorkspaceInBytesOnHost);
    GlobalV::ofs_running << "run cusolvermp 3!!" << std::endl;

    /* Distributed host workspace */
    void *h_sygvdWork = NULL;
    h_sygvdWork = (void *)malloc(sygvdWorkspaceInBytesOnHost);

    /* Distributed device workspace */
    void *d_sygvdWork = NULL;
    cudaMalloc((void **)&d_sygvdWork, sygvdWorkspaceInBytesOnDevice);

    int *d_sygvdInfo = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_sygvdInfo, sizeof(int)));
    checkCudaErrors(cudaMemset(d_sygvdInfo, 0, sizeof(int)));

    /* sync wait for data to arrive to device */
    calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);

    #define MP_TIME_MEASURE 0

    #if MP_TIME_MEASURE
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    #endif

    GlobalV::ofs_running << "run cusolvermp 4!!" << std::endl;

    cusolverMpSygvd(cusolverMpHandle,
                    CUSOLVER_EIG_TYPE_1,
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    this->nFull,   // TODO 这个参数设法应该不对
                    d_A,
                    this->matrix_i,  
                    this->matrix_j,
                    descA,
                    d_B,
                    this->matrix_i,
                    this->matrix_j,
                    descB,
                    d_D,
                    d_Z,
                    this->matrix_i,
                    this->matrix_j,
                    descZ,
                    this->datatype,
                    d_sygvdWork,
                    sygvdWorkspaceInBytesOnDevice,
                    h_sygvdWork,
                    sygvdWorkspaceInBytesOnHost,
                    d_sygvdInfo);
    GlobalV::ofs_running << "run cusolvermp 5!!" << std::endl;

    int h_sygvdInfo = 0;
    checkCudaErrors(cudaMemcpyAsync(&h_sygvdInfo, d_sygvdInfo, sizeof(int), cudaMemcpyDeviceToHost, localStream));
    /* wait for d_sygvdInfo copy */
    checkCudaErrors(cudaStreamSynchronize(localStream));

    #if MP_TIME_MEASURE
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    float seconds = milliseconds / 1000;
    GlobalV::ofs_running << "Execution time: " << seconds << " seconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    #endif
    calStat = cal_stream_sync(cal_comm, localStream);
    assert(calStat == CAL_OK);

    checkCudaErrors(cudaFree(d_sygvdWork));
    checkCudaErrors(cudaFree(d_sygvdInfo));

    free(h_sygvdWork);

    checkCudaErrors(cudaMemcpy((void*)EigenValue, this->d_D, this->nFull * sizeof(outputT), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)EigenVector, this->d_Z, this->narows * this->nacols * sizeof(inputT), cudaMemcpyDeviceToHost));

    return 0;
}
template<typename inputT>
void Diag_CusolverMP_gvd<inputT>::outputParameters()
{
    GlobalV::ofs_running << "nFull: " << this->nFull << std::endl
              << "nev: " << this->nev << std::endl
              << "narows: " << this->narows << std::endl
              << "nacols: " << this->nacols << std::endl
              << "nblk: " << this->nblk << std::endl
              << "lda: " << this->lda << std::endl
              << "nprows: " << this->nprows << std::endl
              << "npcols: " << this->npcols << std::endl
              << "myprow: " << this->myprow << std::endl
              << "mypcol: " << this->mypcol << std::endl
              << "my_global_mpi_id: " << this->my_global_mpi_id << std::endl
              << "mpi_size: " << this->mpi_size << std::endl
              << "matrix_i: " << this->matrix_i << std::endl
              << "matrix_j: " << this->matrix_j << std::endl;
}

template class Diag_CusolverMP_gvd<double>;
template class Diag_CusolverMP_gvd<std::complex<double>>;