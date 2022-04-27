#include "blas.hpp"


double BLAS::Dot(const SystemVector& a, const SystemVector& b)
{
    const auto& size = a.GetSize();
    double result = 0;
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                result += a(i, j, k) * b(i, j, k);
            }
        }
    }
    return result;
}

void BLAS::Residual(const SystemMatrix& A, const SystemVector& x, const SystemVector& b, SystemVector* result)
{
    const auto& size = x.GetSize();

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                (*result)(i, j, k) =
                    b(i, j, k) - A(i, j, k).center * x(i, j, k) -
                    ((i > 0) ? A(i - 1, j, k).right * x(i - 1, j, k) : 0.0) -
                    ((i + 1 < size.x) ? A(i, j, k).right * x(i + 1, j, k) : 0.0) -
                    ((j > 0) ? A(i, j - 1, k).up * x(i, j - 1, k) : 0.0) -
                    ((j + 1 < size.y) ? A(i, j, k).up * x(i, j + 1, k) : 0.0) -
                    ((k > 0) ? A(i, j, k - 1).front * x(i, j, k - 1) : 0.0) -
                    ((k + 1 < size.z) ? A(i, j, k).front * x(i, j, k + 1) : 0.0);
            }
        }
    }
}

void BLAS::AXpY(double a, const SystemVector& x, const SystemVector& y, SystemVector* result)
{
    Vector3<size_t> size = x.GetSize();

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                (*result)(i, j, k) = a * x(i, j, k) + y(i, j, k);
            }
        }
    }
}

double BLAS::L2Norm(const SystemVector& vector)
{
    return std::sqrt(Dot(vector, vector));
}

double BLAS::LInfNorm(const SystemVector& vector)
{
    double absMax = 0;
    for(size_t i = 0; i < vector.GetSize().x; i++)
    {
        for(size_t j = 0; j < vector.GetSize().y; j++)
        {
            for(size_t k = 0 ; k < vector.GetSize().z; k++)
            {
                if(std::abs(vector(i, j, k)) > absMax)
                    absMax = std::abs(vector(i, j, k));
            }
        }
    }
    return std::fabs(absMax);
}