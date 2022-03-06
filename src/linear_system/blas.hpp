#ifndef _BLAS_HPP
#define _BLAS_HPP

#include <cmath>
#include "../common/array3.hpp"
#include "linear_system.hpp"

// TO DO: CUDA version or maybe not well see

class BLAS
{
    public:
        static double Dot(const SystemVector& a, const SystemVector& b);
        static void Residual(const SystemMatrix& A, const SystemVector& x, const SystemVector& b, SystemVector* result);
        static void AXpY(double a, const SystemVector& x, const SystemVector& y, SystemVector* result);
        static double L2Norm(const SystemVector& vector);
        static double LInfNorm(const SystemVector& vector);
};

#endif // BLAS