#include "cuda_jacobi_iteration_solver.hpp"


__global__ void cuda_Relax(double* ACenter, double* ARight, double* AUp, double* AFront, double* x, double* b, double* xTemp, const size_t sizeX, const size_t sizeY, const size_t sizeZ)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + sizeX * (j + sizeY * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < sizeX && j < sizeY && k < sizeZ)
    {
        double r =
                ((i > 0) ? ARight[(i - 1) + sizeX * (j + sizeY * k)] * x[(i - 1) + sizeX * (j + sizeY * k)] : 0.0) +
                ((i + 1 < sizeX) ? ARight[i + sizeX * (j + sizeY * k)] * x[(i + 1) + sizeX * (j + sizeY * k)] : 0.0) +
                ((j > 0) ? AUp[i + sizeX * ((j - 1) + sizeY * k)] * x[i + sizeX * ((j - 1) + sizeY * k)] : 0.0) +
                ((j + 1 < sizeY) ? AUp[i + sizeX * (j + sizeY * k)] * x[i + sizeX * ((j + 1) + sizeY * k)] : 0.0) +
                ((k > 0) ? AFront[i + sizeX * (j + sizeY * (k - 1))] * x[i + sizeX * (j + sizeY * (k - 1))] : 0.0) +
                ((k + 1 < sizeZ) ? AFront[i + sizeX * (j + sizeY * k)] * x[i + sizeX * (j + sizeY * (k + 1))] : 0.0);

        xTemp[idx] = (b[idx] - r) / ACenter[idx];
    }
}


CudaJacobiIterationSolver::CudaJacobiIterationSolver(size_t maxNumberOfIterations, size_t toleranceCheckInterval, double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations), _toleranceCheckInterval(toleranceCheckInterval), _tolerance(tolerance)
{

}

CudaJacobiIterationSolver::~CudaJacobiIterationSolver()
{

}
#include <iostream>
void CudaJacobiIterationSolver::Solve(LinearSystem* system)
{
    const auto size = system->x.GetSize();
    Initialize(system);
    double* x = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    cudaMemcpy(x, _d_x, size.x * size.y * size.z *sizeof(double), cudaMemcpyDeviceToHost);
    free(x);

    for(size_t i = 0; i < _maxNumberOfIterations; i++)
    {
        _iteration = i;
        Relax(size);
        if(i != 0 && i % _toleranceCheckInterval == 0)
        {
            if(CalculateTolerance(size) < _tolerance)
            {
                break;
            }
        }
    }

    FromDeviceToHost(system);
    FreeDeviceMemory();
}

void CudaJacobiIterationSolver::Relax(const Vector3<size_t> size)
{
    int threadsInX = 8;
    int threadsInY = 8;
    int threadsInZ = 8;

    int blocksInX = (int)std::ceil(((double)size.x) / threadsInX);
    int blocksInY = (int)std::ceil(((double)size.y) / threadsInY);
    int blocksInZ = (int)std::ceil(((double)size.z) / threadsInZ);

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

    cuda_Relax<<<dimGrid, dimBlock>>>(_d_ACenter, _d_ARight, _d_AUp, _d_AFront, _d_x, _d_b, _d_xTemp, size.x, size.y, size.z);

    cudaDeviceSynchronize();
    const unsigned int vectorSize = (unsigned int)size.x * (unsigned int)size.y * (unsigned int)size.z;
    cudaMemcpy(_d_tmp, _d_xTemp, vectorSize * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(_d_xTemp, _d_x, vectorSize * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(_d_x, _d_tmp, vectorSize * sizeof(double), cudaMemcpyDeviceToDevice);
}

double CudaJacobiIterationSolver::CalculateTolerance(const Vector3<size_t> size)
{
    CUDA_BLAS::Residual(_d_ACenter, _d_ARight, _d_AUp, _d_AFront, _d_x, _d_b, _d_residual, size);
    return CUDA_BLAS::L2Norm(_d_residual, size);
}

void CudaJacobiIterationSolver::Initialize(LinearSystem* system)
{
    const auto& size = system->x.GetSize();
    const unsigned int vectorSize = (unsigned int)size.x * (unsigned int)size.y * (unsigned int)size.z;
    double* x, *b, *ACenter, *ARight, *AUp, *AFront, *residual, *xTemp;
    x = (double*)malloc(vectorSize * sizeof(double));
    b = (double*)malloc(vectorSize * sizeof(double));
    ACenter = (double*)malloc(vectorSize * sizeof(double));
    ARight = (double*)malloc(vectorSize * sizeof(double));
    AUp = (double*)malloc(vectorSize * sizeof(double));
    AFront = (double*)malloc(vectorSize * sizeof(double));
    residual = (double*)malloc(vectorSize * sizeof(double));
    xTemp = (double*)malloc(vectorSize * sizeof(double));

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                x[i + size.x * (j + size.y * k)] = system->x(i, j, k);
                b[i + size.x * (j + size.y * k)] = system->b(i, j, k);
                ACenter[i + size.x * (j + size.y * k)] = system->A(i, j, k).center;
                ARight[i + size.x * (j + size.y * k)] = system->A(i, j, k).right;
                AUp[i + size.x * (j + size.y * k)] = system->A(i, j, k).up;
                AFront[i + size.x * (j + size.y * k)] = system->A(i, j, k).front;
                residual[i + size.x * (j + size.y * k)] = 0;
                xTemp[i + size.x * (j + size.y * k)] = 0;
            }   
        }
    }

    cudaMalloc((void **)&_d_x, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_b, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_ACenter, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_ARight, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_AUp, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_AFront, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_residual, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_xTemp, vectorSize * sizeof(double));
    cudaMalloc((void **)&_d_tmp, vectorSize * sizeof(double));

    cudaMemcpy(_d_x, x, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_b, b, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_ACenter, ACenter, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_ARight, ARight, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_AUp, AUp, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_AFront, AFront, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_residual, residual, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_xTemp, xTemp, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_tmp, xTemp, vectorSize * sizeof(double), cudaMemcpyHostToDevice);

    free(x);
    free(b);
    free(ACenter);
    free(ARight);
    free(AFront);
    free(residual);
    free(xTemp);
}

void CudaJacobiIterationSolver::FromDeviceToHost(LinearSystem* system)
{
    const auto size = system->x.GetSize();
    const unsigned int vectorSize = (unsigned int)size.x * (unsigned int)size.y * (unsigned int)size.z;
    double *x, *b, *ACenter, *ARight, *AUp, *AFront;
    x = (double*)malloc(vectorSize * sizeof(double));
    b = (double*)malloc(vectorSize * sizeof(double));
    ACenter = (double*)malloc(vectorSize * sizeof(double));
    ARight = (double*)malloc(vectorSize * sizeof(double));
    AUp = (double*)malloc(vectorSize * sizeof(double));
    AFront = (double*)malloc(vectorSize * sizeof(double));

    cudaMemcpy(x, _d_x, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, _d_b, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ACenter, _d_ACenter, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ARight, _d_ARight, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(AUp, _d_AUp, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(AFront, _d_AFront, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                system->x(i, j, k) = x[i + size.x * (j + size.y * k)];
                system->b(i, j, k) = b[i + size.x * (j + size.y * k)];
                system->A(i, j, k).center = ACenter[i + size.x * (j + size.y * k)];
                system->A(i, j, k).right = ARight[i + size.x * (j + size.y * k)];
                system->A(i, j, k).up = AUp[i + size.x * (j + size.y * k)];
                system->A(i, j, k).front = AFront[i + size.x * (j + size.y * k)];
            }   
        }
    }
    free(x);
    free(b);
    free(ACenter);
    free(ARight);
    free(AUp);
    free(AFront);
}

void CudaJacobiIterationSolver::FreeDeviceMemory()
{
    cudaFree(_d_x);
    cudaFree(_d_b);
    cudaFree(_d_ACenter);
    cudaFree(_d_ARight);
    cudaFree(_d_AUp);
    cudaFree(_d_AFront);
    cudaFree(_d_residual);
    cudaFree(_d_xTemp);
    cudaFree(_d_tmp);
}
