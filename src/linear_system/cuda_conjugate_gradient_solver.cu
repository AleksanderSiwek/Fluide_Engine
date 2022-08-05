#include "cuda_conjugate_gradient_solver.hpp"
#include "blas.hpp"

CudaConjugateGradientSolver::CudaConjugateGradientSolver(size_t maxNumberOfIterations, double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations), _tolerance(tolerance), _wasMemoryAllocatedOnDevice(false)
{

}

CudaConjugateGradientSolver::~CudaConjugateGradientSolver()
{
    if(_wasMemoryAllocatedOnDevice)
    {
        FreeDeviceMemory();
    }
}

void CudaConjugateGradientSolver::Solve(LinearSystem* system)
{
    const auto& size = system->x.GetSize();
    CUDA_Int3 cudaSize = Vector3SizeToCUDA_Int3(size);
    FromHostToDevice(system);

    const int threadsInX = 4;
    const int threadsInY = 4;
    const int threadsInZ = 4;

    const int blocksInX = (int)std::ceil(((double)size.x) / threadsInX);
    const int blocksInY = (int)std::ceil(((double)size.y) / threadsInY);
    const int blocksInZ = (int)std::ceil(((double)size.z) / threadsInZ);

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

    CUDA_FillArray3<<<dimGrid, dimBlock>>>(_d_r, 0, cudaSize);
    CUDA_FillArray3<<<dimGrid, dimBlock>>>(_d_d, 0, cudaSize);
    CUDA_FillArray3<<<dimGrid, dimBlock>>>(_d_q, 0, cudaSize);
    CUDA_FillArray3<<<dimGrid, dimBlock>>>(_d_s, 0, cudaSize);
    CUDA_FillArray3<<<dimGrid, dimBlock>>>(_d_tmp, 0, cudaSize);

    CUDA_BLAS::CUDA_Residual<<<dimGrid, dimBlock>>>(_d_ACenter, _d_ARight, _d_AUp, _d_AFront, _d_x, _d_b, _d_r, size.x, size.y, size.z);

    Preconditioner(_d_r, _d_d, cudaSize, dimGrid, dimBlock);

    double sigmaNew = CUDA_BLAS::Dot(_d_r, _d_d, size);
    size_t iteration = 0 ;
    bool trigger = false;
    while(sigmaNew > _tolerance * _tolerance && iteration < _maxNumberOfIterations)
    {
        CUDA_BLAS::CUDA_MatrixVectorMultiplication<<<dimGrid, dimBlock>>>(_d_ACenter, _d_ARight, _d_AUp, _d_AFront, _d_d, _d_q, cudaSize);
        double alpha = sigmaNew / CUDA_BLAS::Dot(_d_d, _d_q, size);
        cudaMemcpy(_d_tmp, _d_x, size.x * size.y * size.z * sizeof(double), cudaMemcpyDeviceToDevice);
        CUDA_BLAS::CUDA_AXpY<<<dimGrid, dimBlock>>>(alpha, _d_d, _d_tmp, _d_x, cudaSize);

        if(trigger || (iteration % 50 == 0 && iteration > 0))
        {
            CUDA_BLAS::CUDA_Residual<<<dimGrid, dimBlock>>>(_d_ACenter, _d_ARight, _d_AUp, _d_AFront, _d_x, _d_b, _d_r, size.x, size.y, size.z);
            trigger = false;
        }
        else
        {
            cudaMemcpy(_d_tmp, _d_r, size.x * size.y * size.z * sizeof(double), cudaMemcpyDeviceToDevice);
            CUDA_BLAS::CUDA_AXpY<<<dimGrid, dimBlock>>>(-alpha, _d_q, _d_tmp, _d_r, cudaSize);
        }

        Preconditioner(_d_r, _d_s, cudaSize, dimGrid, dimBlock);

        double sigmaOld = sigmaNew;
        sigmaNew = CUDA_BLAS::Dot(_d_r, _d_s, size);
        if(sigmaNew > sigmaOld)
        {
            trigger = true;
        }

        double beta = sigmaNew / sigmaOld;
        cudaMemcpy(_d_tmp, _d_d, size.x * size.y * size.z * sizeof(double), cudaMemcpyDeviceToDevice);
        CUDA_BLAS::CUDA_AXpY<<<dimGrid, dimBlock>>>(beta, _d_tmp, _d_s, _d_d, cudaSize);
        iteration++;
    }

    FromDeviceToHost(system);
}


void CudaConjugateGradientSolver::Preconditioner(double* a, double* b, CUDA_Int3 size, dim3 dimGrid, dim3 dimBlock)
{
    cudaMemcpy(b, a, size.x * size.y * size.z * sizeof(double), cudaMemcpyDeviceToDevice);
}

void CudaConjugateGradientSolver::InitializeSolver(const Vector3<size_t>& size)
{

}

void CudaConjugateGradientSolver::FromHostToDevice(LinearSystem* system)
{
    const auto& size = system->x.GetSize();
    const unsigned int vectorSize = (unsigned int)size.x * (unsigned int)size.y * (unsigned int)size.z;
    double *ACenter, *ARight, *AUp, *AFront, *b;
    ACenter = (double*)malloc(vectorSize * sizeof(double));
    ARight = (double*)malloc(vectorSize * sizeof(double));
    AUp = (double*)malloc(vectorSize * sizeof(double));
    AFront = (double*)malloc(vectorSize * sizeof(double));
    b = (double*)malloc(vectorSize * sizeof(double));

    parallel_utils::ForEach3(size.x, size.y, size.z, [&](size_t i, size_t j, size_t k)
    {
        ACenter[i + size.x * (j + size.y * k)] = system->A(i, j, k).center;
        ARight[i + size.x * (j + size.y * k)] = system->A(i, j, k).right;
        AUp[i + size.x * (j + size.y * k)] = system->A(i, j, k).up;
        AFront[i + size.x * (j + size.y * k)] = system->A(i, j, k).front;
        b[i + size.x * (j + size.y * k)] = system->b(i, j, k);
    });

    // TO DO: check if last size matches current size 
    if(!_wasMemoryAllocatedOnDevice)
    {
       AllocateMemoryOnDevice(vectorSize);
       _wasMemoryAllocatedOnDevice = true;
    }

    cudaMemcpy(_d_ACenter, ACenter, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_ARight, ARight, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_AUp, AUp, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_AFront, AFront, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_b, b, vectorSize * sizeof(double), cudaMemcpyHostToDevice);

    free(ACenter);
    free(ARight);
    free(AUp);
    free(AFront);
    free(b);
}

void CudaConjugateGradientSolver::FromDeviceToHost(LinearSystem* system)
{
    const auto size = system->x.GetSize();
    const unsigned int vectorSize = (unsigned int)size.x * (unsigned int)size.y * (unsigned int)size.z;
    double *x;
    x = (double*)malloc(vectorSize * sizeof(double));

    cudaMemcpy(x, _d_x, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);

    parallel_utils::ForEach3(size.x, size.y, size.z, [&](size_t i, size_t j, size_t k)
    {
        system->x(i, j, k) = x[i + size.x * (j + size.y * k)];
    });

    free(x);
}

void CudaConjugateGradientSolver::AllocateMemoryOnDevice(size_t vectorSize)
{
    cudaMalloc((void **)&_d_x, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_b, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_ACenter, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_ARight, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_AUp, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_AFront, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_r, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_d, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_q, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_s, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_tmp, vectorSize * sizeof(double));
}

void CudaConjugateGradientSolver::FreeDeviceMemory()
{    
    cudaFree(_d_x);
    cudaFree(_d_b);
    cudaFree(_d_ACenter);
    cudaFree(_d_ARight);
    cudaFree(_d_AUp);
    cudaFree(_d_AFront);
    cudaFree(_d_r);
    cudaFree(_d_d);
    cudaFree(_d_q);
    cudaFree(_d_s);
    cudaFree(_d_tmp);
}