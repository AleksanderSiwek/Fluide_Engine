#include <gtest/gtest.h>
#include "../src/linear_system./blas.hpp"
#include "../src/linear_system./cuda_blas.hpp"
#include "../src/linear_system/linear_system.hpp"

#include <iostream>

TEST(CUDA_BLASTest, Dot_test)
{
    Vector3<size_t> size(20, 20, 20);
    SystemVector v1(size, 2);
    SystemVector v2(size, 3);
    v1(2, 2, 2) = 62;
    v1(1, 2, 2) = 656;
    v1(0, 0, 2) = 79;
    v2(0, 1, 0) = 6;
    v2(1, 1, 1) = 9;

    double ret = CUDA_BLAS::Dot(v1, v2);
    double ret_2 = BLAS::Dot(v1, v2);
    EXPECT_EQ(ret_2, ret);
}

TEST(CUDA_BLASTest, Residual_test)
{
    struct LinearSystemMatrixRow initialValue = {2, 3, 5, 4};
    SystemMatrix A(2, 2, 2, initialValue);
    SystemVector v1(2, 2, 2, 2);
    SystemVector v2(2, 2, 2, 3);
    SystemVector result(2, 2, 2, 3);
    SystemVector result_2(2, 2, 2, 3);
    CUDA_BLAS::Residual(A, v1, v2, &result);
    BLAS::Residual(A, v1, v2, &result_2);
    EXPECT_EQ(true, result.IsEqual(result_2));
}

TEST(CUDA_BLASTest, L2Norm_test)
{
    SystemVector x(2, 2, 2, 2);
    EXPECT_EQ(sqrt(32), CUDA_BLAS::L2Norm(x));
}

TEST(CUDA_BLASTest, CUDA_AXpY)
{
    Vector3<size_t> size(3, 3, 3);
    SystemVector x(size, 2);
    SystemVector y(size, 3);
    SystemVector result(size, 2);
    x(0, 0, 0) = 21;
    x(0, 0, 1) = 27;
    x(1, 2, 1) = 69;
    y(1, 2, 1) = 29;
    y(1, 2, 1) = 99;
    y(1, 2, 1) = 79;
    double val = 5;

    CUDA_Int3 cudaSize = Vector3SizeToCUDA_Int3(size);
    size_t vectorSize = size.x * size.y * size.z;

    BLAS::AXpY(val, x, y, &result);

    double *d_x, *d_y, *d_result;
    double* h_result = (double*)malloc(vectorSize * sizeof(double));
    cudaMalloc((void**)&d_x, vectorSize * sizeof(double));
    cudaMalloc((void**)&d_y, vectorSize * sizeof(double));
    cudaMalloc((void**)&d_result, vectorSize * sizeof(double));

    cudaMemcpy(d_x, &(x.GetRawData())[0], vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &(y.GetRawData())[0], vectorSize * sizeof(double), cudaMemcpyHostToDevice);

    const int threadsInX = 4;
    const int threadsInY = 4;
    const int threadsInZ = 4;

    const int blocksInX = (int)std::ceil(((double)size.x) / threadsInX);
    const int blocksInY = (int)std::ceil(((double)size.y) / threadsInY);
    const int blocksInZ = (int)std::ceil(((double)size.z) / threadsInZ);

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

    CUDA_BLAS::CUDA_AXpY<<<dimGrid, dimBlock>>>(val, d_x, d_y, d_result, cudaSize);

    cudaMemcpy(h_result, d_result, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < vectorSize; i++)
    {
        EXPECT_NEAR(h_result[i], result.GetRawData()[i], 1e-6);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    free(h_result);
}

TEST(CUDA_BLASTest, CUDA_MatrixVectorMultiplication)
{
    Vector3<size_t> size(3, 3, 3);
    struct LinearSystemMatrixRow initialValue = {2, 3, 5, 4};
    SystemMatrix A(size, initialValue);
    SystemVector v(size);
    SystemVector result(size, 2);
    A(0, 0, 0).center = 69;
    A(0, 0, 1).up = 42;
    A(0, 2, 0).right = 76;
    A(0, 2, 2).front = 32;
    A(1, 1, 1).up = 22;
    A(2, 1, 0).front = 33;
    A(2, 2, 2).center = 44;
    v(1, 1, 1) = 55;
    v(1, 1, 2) = 55;
    CUDA_Int3 cudaSize = Vector3SizeToCUDA_Int3(size);
    size_t vectorSize = size.x * size.y * size.z;

    BLAS::MatrixVectorMultiplication(A, v, &result);

    double *d_AC, *d_AU, *d_AF, *d_AR, *d_v, *d_result;
    double* h_result = (double*)malloc(vectorSize * sizeof(double));
    double* h_AC = (double*)malloc(vectorSize * sizeof(double));
    double* h_AU = (double*)malloc(vectorSize * sizeof(double));
    double* h_AF = (double*)malloc(vectorSize * sizeof(double));
    double* h_AR = (double*)malloc(vectorSize * sizeof(double));
    cudaMalloc((void**)&d_AC, vectorSize * sizeof(double));
    cudaMalloc((void**)&d_AU, vectorSize * sizeof(double));
    cudaMalloc((void**)&d_AF, vectorSize * sizeof(double));
    cudaMalloc((void**)&d_AR, vectorSize * sizeof(double));
    cudaMalloc((void**)&d_v, vectorSize * sizeof(double));
    cudaMalloc((void**)&d_result, vectorSize * sizeof(double));

    A.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        const auto& row = A(i, j, k);
        size_t idx = i + size.x * (j + size.y * k);
        std::cout << "center: " << row.center << " up: " << row.up << " front: " << row.front << " right: " << row.right << "\n";
        h_AC[idx] = row.center;
        h_AU[idx] = row.up;
        h_AF[idx] = row.front;
        h_AR[idx] = row.right;
    });

    cudaMemcpy(d_v, &(v.GetRawData())[0], vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AC, h_AC, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AU, h_AU, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AF, h_AF, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AR, h_AR, vectorSize * sizeof(double), cudaMemcpyHostToDevice);


    const int threadsInX = 4;
    const int threadsInY = 4;
    const int threadsInZ = 4;

    const int blocksInX = (int)std::ceil(((double)size.x) / threadsInX);
    const int blocksInY = (int)std::ceil(((double)size.y) / threadsInY);
    const int blocksInZ = (int)std::ceil(((double)size.z) / threadsInZ);

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

    CUDA_BLAS::CUDA_MatrixVectorMultiplication<<<dimGrid, dimBlock>>>(d_AC, d_AR, d_AU, d_AF, d_v, d_result, cudaSize);

    cudaMemcpy(h_result, d_result, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < vectorSize; i++)
    {
        EXPECT_NEAR(h_result[i], result.GetRawData()[i], 1e-6);
        std::cout << "EX: " << result.GetRawData()[i] << "  EQ: " << h_result[i] << "\n";
    }

    cudaFree(d_AC);
    cudaFree(d_AU);
    cudaFree(d_AF);
    cudaFree(d_AR);
    cudaFree(d_v);
    cudaFree(d_result);
    free(h_AC);
    free(h_AU);
    free(h_AF);
    free(h_AR);
    free(h_result);
}

