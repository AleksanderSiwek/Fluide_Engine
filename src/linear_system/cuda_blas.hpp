#ifndef _CUDA_BLAS_HPP
#define _CUDA_BLAS_HPP

#include <cmath>

#include "../common/vector3.hpp"
#include "../common/array3.hpp"
#include "../linear_system/linear_system.hpp"


class CUDA_BLAS
{
    public:
        static double Dot(double* a, double* b, Vector3<size_t> size);
        static double Dot(Array3<double>& a, Array3<double>& b);
        static void Residual(double* ACenter, double* ARight, double* AUp, double* AFront, double* x, double* b, double* result, Vector3<size_t> size);
        static void Residual(SystemMatrix& A, const SystemVector& x, const SystemVector& b, SystemVector* result);
        static double L2Norm(double* vector, Vector3<size_t> size);
        static double L2Norm(Array3<double>& a);
};

#endif // _CUDA_BLAS_HPP