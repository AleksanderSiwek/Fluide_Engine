#include "blas.hpp"


double BLAS::Dot(const Vector3<double>& a, const Vector3<double>& b)
{
    return a.Dot(b);
}

void BLAS::Residual(double a, const Vector3<double> x, const Vector3<double> y, Vector3<double>* result)
{
    (*result) = a * x - y;
}

void BLAS::AXpY(double a, const Vector3<double> x, const Vector3<double> y, Vector3<double>* result)
{
    (*result) = a * x + y;
}

double BLAS::L2Norm(const Vector3<double>& vector)
{
    return std::sqrt(vector.Dot(vector));
}

double BLAS::LInfNorm(const Vector3<double>& vector)
{
    return std::fabs(vector.AbsMax());
}