#include "scalar_grid3.hpp"


ScalarGrid3::ScalarGrid3(size_t width, size_t height, size_t depth, const double& initailValue, Vector3<double> origin, Vector3<double> gridSpacing)
    : Array3<double>(width, height, depth, initailValue), Grid3D(origin, gridSpacing)
{

}

ScalarGrid3::ScalarGrid3(const Vector3<size_t>& size, const double& initailValue, Vector3<double> origin, Vector3<double> gridSpacing)
    : Array3<double>(size, initailValue), Grid3D(origin, gridSpacing)
{

}

ScalarGrid3::ScalarGrid3(const ScalarGrid3& grid)
    : Array3<double>(grid), Grid3D(grid)
{

}

ScalarGrid3::~ScalarGrid3()
{

}

double ScalarGrid3::Sample(const Vector3<double>& position) const
{
    size_t i, j, k;
    double factorX, factorY, factorZ ;
    i = j = k = 0;
    factorX = factorY = factorZ = 0;

    Vector3<double> normalizedPoistion = (position - _origin) / _gridSpacing;
    Vector3<size_t> size = GetSize();

    GetBarycentric<double>(normalizedPoistion.x, 0, size.x - 1, &i, &factorX);
    GetBarycentric<double>(normalizedPoistion.x, 0, size.x - 1, &i, &factorX);
    GetBarycentric<double>(normalizedPoistion.x, 0, size.x - 1, &i, &factorX);

    size_t ip1 = std::min(i + 1, size.x - 1);
    size_t jp1 = std::min(j + 1, size.y - 1);
    size_t kp1 = std::min(k + 1, size.z - 1);

    return Trilerp<double, double>( GetElement(i, j, k),
                                    GetElement(ip1, j, k),
                                    GetElement(i, jp1, k),
                                    GetElement(ip1, jp1, k),
                                    GetElement(i, j, kp1),
                                    GetElement(ip1, j, kp1),
                                    GetElement(i, jp1, kp1),
                                    GetElement(ip1, jp1, kp1),
                                    factorX,
                                    factorY,
                                    factorZ);
}

Vector3<double> ScalarGrid3::Gradient(const Vector3<double>& position) const
{
    // TO DO
    return 0;
}

double ScalarGrid3::Laplacian(const Vector3<double>& position) const
{
    // TO DO
    return 0;
}
