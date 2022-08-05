#ifndef _BLAS_HPP
#define _BLAS_HPP

#include <cmath>

#include "linear_system.hpp"
#include "../common/array3.hpp"


class BLAS
{
    public:
        static double Dot(const SystemVector& a, const SystemVector& b);
        static void Residual(const SystemMatrix& A, const SystemVector& x, const SystemVector& b, SystemVector* result);
        static void AXpY(double a, const SystemVector& x, const SystemVector& y, SystemVector* result);
        static double L2Norm(const SystemVector& vector);
        static double LInfNorm(const SystemVector& vector);
        static void MatrixVectorMultiplication(const SystemMatrix& a, const SystemVector& b, SystemVector* result);
};

#endif // BLAS