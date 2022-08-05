#ifndef _CUDA_BLAS_HPP
#define _CUDA_BLAS_HPP

#include <cmath>
#include <cuda_runtime.h>

#include "../common/cuda_array_utils.hpp"
#include "../common/vector3.hpp"
#include "../common/array3.hpp"
#include "../linear_system/linear_system.hpp"


namespace CUDA_BLAS
{
    double Dot(double* a, double* b, Vector3<size_t> size);
    double Dot(Array3<double>& a, Array3<double>& b);
    void Residual(double* ACenter, double* ARight, double* AUp, double* AFront, double* x, double* b, double* result, Vector3<size_t> size);
    void Residual(SystemMatrix& A, const SystemVector& x, const SystemVector& b, SystemVector* result);
    double L2Norm(double* vector, Vector3<size_t> size);
    double L2Norm(Array3<double>& a);
    __global__ void CUDA_DOT(double* a, double* b, double* result, const size_t size);
    __global__ void CUDA_Residual(double* ACenter, double* ARight, double* AUp, double* AFront, double* x, double* b, double* result, const size_t sizeX, const size_t sizeY, const size_t sizeZ);
    __global__ void CUDA_MatrixVectorMultiplication(double* ACenter, double* ARight, double* AUp, double* AFront, double* vector, double* result, CUDA_Int3 size);
    __global__ void CUDA_AXpY(double a, double* x, double* y, double* result, CUDA_Int3 size);
}

#endif // _CUDA_BLAS_HPP