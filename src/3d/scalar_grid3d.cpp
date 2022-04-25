#include "scalar_grid3d.hpp"


ScalarGrid3D::ScalarGrid3D(size_t width, size_t height, size_t depth, const double& initailValue, Vector3<double> origin, Vector3<double> gridSpacing)
    : Array3<double>(width, height, depth, initailValue), Grid3D(origin, gridSpacing)
{

}

ScalarGrid3D::ScalarGrid3D(const Vector3<size_t>& size, const double& initailValue, Vector3<double> origin, Vector3<double> gridSpacing)
    : Array3<double>(size, initailValue), Grid3D(origin, gridSpacing)
{

}

ScalarGrid3D::ScalarGrid3D(const ScalarGrid3D& grid)
    : Array3<double>(grid), Grid3D(grid)
{

}

ScalarGrid3D::~ScalarGrid3D()
{

}

double ScalarGrid3D::Sample(const Vector3<double>& position) const
{
    int i, j, k;
    double factorX, factorY, factorZ ;
    i = j = k = 0;
    factorX = factorY = factorZ = 0;

    Vector3<double> normalizedPoistion = (position - _origin) / _gridSpacing;
    const auto& size = GetSize();
    int sizeX = static_cast<int>(size.x);
    int sizeY = static_cast<int>(size.y);
    int sizeZ = static_cast<int>(size.z);

    GetBarycentric<double>(normalizedPoistion.x, 0, sizeX - 1, &i, &factorX);
    GetBarycentric<double>(normalizedPoistion.y, 0, sizeY - 1, &j, &factorY);
    GetBarycentric<double>(normalizedPoistion.z, 0, sizeZ - 1, &k, &factorZ);

    size_t ip1 = std::min(i + 1, sizeX - 1);
    size_t jp1 = std::min(j + 1, sizeY - 1);
    size_t kp1 = std::min(k + 1, sizeZ - 1);

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

Vector3<double> ScalarGrid3D::Gradient(const Vector3<double>& position) const
{
    // TO DO
    return 0;
}

double ScalarGrid3D::Laplacian(const Vector3<double>& position) const
{
    // TO DO
    return 0;
}
