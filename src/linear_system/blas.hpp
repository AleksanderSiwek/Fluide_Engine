#ifndef _BLAS_HPP
#define _BLAS_HPP

#include <cmath>
#include "../common/array3.hpp"
#include "linear_system.hpp"

// TO DO: CUDA version or maybe not well see

class BLAS
{
    public:
        static double Dot(const Vector3<double>& a, const Vector3<double>& b);
        static void Residual(double a, const Vector3<double> x, const Vector3<double> y, Vector3<double>* result);
        static void AXpY(double a, const Vector3<double> x, const Vector3<double> y, Vector3<double>* result);
        static double L2Norm(const Vector3<double>& vector);
        static double LInfNorm(const Vector3<double>& vector);
};

#endif // BLAS