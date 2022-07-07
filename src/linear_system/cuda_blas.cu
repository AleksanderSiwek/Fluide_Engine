#include "cuda_blas.hpp"

#define BLOCK_SIZE 512


__global__ void cuda_DOT(double* a, double* b, double* result, const size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sum[BLOCK_SIZE];
    if(idx >= size) 
    {
        sum[threadIdx.x] = 0;
    }
    else 
    {
        sum[threadIdx.x] = a[idx] * b[idx];
    }

    __syncthreads();

    for(int n = BLOCK_SIZE / 2; n > 0; n /= 2)
    {
        if(threadIdx.x < n) 
        {
            sum[threadIdx.x] += sum[threadIdx.x + n];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) 
    {
        result[blockIdx.x] = sum[0];
    }
}

__global__ void cuda_Residual(double* ACenter, double* ARight, double* AUp, double* AFront, double* x, double* b, double* result, const size_t sizeX, const size_t sizeY, const size_t sizeZ)
{
    unsigned int blockIdxz = __float2uint_rd(blockIdx.y * 1/8);
    unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz, 8);
    int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int j = __umul24(blockIdxy, blockDim.y) + threadIdx.y;
    int k = __umul24(blockIdxz, blockDim.z) + threadIdx.z;
    unsigned int idx = i + sizeX * (j + sizeY * k);

    if(i < 0 || j < 0 || k < 0 || i >= sizeX || j >= sizeY || k >= sizeZ)
        return;

    result[idx] =
                    b[idx] - ACenter[idx] * x[idx] -
                    ((i > 0) ? ARight[(i - 1) + sizeX * (j + sizeY * k)] * x[(i - 1) + sizeX * (j + sizeY * k)] : 0.0) -
                    ((i + 1 < sizeX) ? ARight[i + sizeX * (j + sizeY * k)] * x[(i + 1) + sizeX * (j + sizeY * k)] : 0.0) -
                    ((j > 0) ? AUp[i + sizeX * ((j - 1) + sizeY * k)] * x[i + sizeX * ((j - 1) + sizeY * k)] : 0.0) -
                    ((j + 1 < sizeY) ? AUp[i + sizeX * (j + sizeY * k)] * x[i + sizeX * ((j + 1) + sizeY * k)] : 0.0) -
                    ((k > 0) ? AFront[i + sizeX * (j + sizeY * (k - 1))] * x[i + sizeX * (j + sizeY * (k - 1))] : 0.0) -
                    ((k + 1 < sizeZ) ? AFront[i + sizeX * (j + sizeY * k)] * x[i + sizeX * (j + sizeY * (k + 1))] : 0.0);
}

double CUDA_BLAS::Dot(double* a, double* b, Vector3<size_t> size)
{
    double result = 0;
    double* h_result;
    double* d_result;

    unsigned int vectorLength = (int)size.x * (int)size.y * (int)size.z;
    unsigned int gridSize = vectorLength / BLOCK_SIZE + 1 - (vectorLength % BLOCK_SIZE == 0);
    dim3 dimGrid = dim3(gridSize);
    dim3 dimBlock = dim3(BLOCK_SIZE);

    h_result = (double*)malloc(gridSize* sizeof(double));
    cudaMalloc((void **)&d_result, gridSize * sizeof(double));

    cuda_DOT<<<dimGrid, dimBlock>>>(a, b, d_result, vectorLength);
    cudaMemcpy(h_result, d_result, gridSize * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < gridSize; i++) 
    {
        result += h_result[i];
    }

    cudaFree(d_result);
    free(h_result);
    return result;
}

double CUDA_BLAS::Dot(Array3<double>& a, Array3<double>& b)
{
    const auto& size = a.GetSize();
    double *h_a, *h_b;
    h_a = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    h_b = (double*)malloc(size.x * size.y * size.z * sizeof(double));

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                h_a[i + size.x * (j + size.y * k)] = a(i, j, k);
                h_b[i + size.x * (j + size.y * k)] = b(i, j, k);
            }   
        }
    }

    double *d_a, *d_b;
    cudaMalloc((void **)&d_a, size.x * size.y * size.z * sizeof(double));
    cudaMalloc((void **)&d_b, size.x * size.y * size.z * sizeof(double));

    cudaMemcpy(d_a, h_a, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);

    double result = Dot(d_a, d_b, size);

    cudaFree(d_a);
    cudaFree(d_b);
    return result;
}

void CUDA_BLAS::Residual(double* ACenter, double* ARight, double* AUp, double* AFront, double* x, double* b, double* result, Vector3<size_t> size)
{ 
    int threadsInX = 8;
    int threadsInY = 8;
    int threadsInZ = 8;

    int blocksInX = ((int)size.x + threadsInX-1) / threadsInX;
    int blocksInY = ((int)size.y + threadsInY-1) / threadsInY;
    int blocksInZ = ((int)size.z + threadsInZ-1) / threadsInZ;

    dim3 dimGrid = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);
    cuda_Residual<<<dimGrid, dimBlock>>>(ACenter, ARight, AUp, AFront, x, b, result, size.x, size.y, size.z);
}

void CUDA_BLAS::Residual(SystemMatrix& A, const SystemVector& x, const SystemVector& b, SystemVector* result)
{
    const auto& size = A.GetSize();
    double *h_ACenter, *h_ARight, *h_AUp, *h_AFront, *h_x, *h_b, *h_result;
    h_ACenter = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    h_ARight = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    h_AUp = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    h_AFront = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    h_x = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    h_b = (double*)malloc(size.x * size.y * size.z * sizeof(double));
    h_result = (double*)malloc(size.x * size.y * size.z * sizeof(double));

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                h_ACenter[i + size.x * (j + size.y * k)] = A(i, j, k).center;
                h_ARight[i + size.x * (j + size.y * k)] = A(i, j, k).right;
                h_AUp[i + size.x * (j + size.y * k)] = A(i, j, k).up;
                h_AFront[i + size.x * (j + size.y * k)] = A(i, j, k).front;
                h_x[i + size.x * (j + size.y * k)] = x(i, j, k);
                h_b[i + size.x * (j + size.y * k)] = b(i, j, k);
            }   
        }
    }

    double *d_ACenter, *d_ARight, *d_AUp, *d_AFront, *d_x, *d_b, *d_result;
    cudaMalloc((void **)&d_ACenter, size.x * size.y * size.z * sizeof(double));
    cudaMalloc((void **)&d_ARight, size.x * size.y * size.z * sizeof(double));
    cudaMalloc((void **)&d_AUp, size.x * size.y * size.z * sizeof(double));
    cudaMalloc((void **)&d_AFront, size.x * size.y * size.z * sizeof(double));
    cudaMalloc((void **)&d_x, size.x * size.y * size.z * sizeof(double));
    cudaMalloc((void **)&d_b, size.x * size.y * size.z * sizeof(double));
    cudaMalloc((void **)&d_result, size.x * size.y * size.z * sizeof(double));

    cudaMemcpy(d_ACenter, h_ACenter, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ARight, h_ARight, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AUp, h_AUp, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AFront, h_AFront, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size.x * size.y * size.z * sizeof(double), cudaMemcpyHostToDevice);

    Residual(d_ACenter, d_ARight, d_AUp, d_AFront, d_x, d_b, d_result, size);

    cudaMemcpy(h_result, d_result, size.x * size.y * size.z * sizeof(double), cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < A.GetSize().x; i++)
    {
        for(size_t j = 0; j < A.GetSize().y; j++)
        {
            for(size_t k = 0; k < A.GetSize().z; k++)
            {
                (*result)(i, j, k) = h_result[i + size.x * (j + size.y * k)];
            }   
        }  
    }
    result->GetRawData().assign(h_result, h_result + size.x * size.y * size.z);
   
    cudaFree(d_ACenter);
    cudaFree(d_ARight);
    cudaFree(d_AUp);
    cudaFree(d_AFront);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_ACenter);
    free(h_ARight);
    free(h_AUp);
    free(h_AFront);
    free(h_x);
    free(h_b);
    free(h_result);
}

double CUDA_BLAS::L2Norm(double* vector, Vector3<size_t> size)
{
    return std::sqrt(Dot(vector, vector, size));
}

double CUDA_BLAS::L2Norm(Array3<double>& a)
{
    return std::sqrt(Dot(a, a));
}