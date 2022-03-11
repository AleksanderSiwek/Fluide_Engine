#include "scalar_grid3.hpp"


ScalarGrid3::ScalarGrid3(size_t width, size_t height, size_t depth, const double& initailValue, Vector3<double> origin)
    : Array3<double>(width, height, depth, initailValue), _origin(origin)
{

}

ScalarGrid3::ScalarGrid3(const Vector3<size_t>& size, const double& initailValue, Vector3<double> origin)
    : Array3<double>(size, initailValue), _origin(origin)
{

}

ScalarGrid3::ScalarGrid3(const ScalarGrid3& grid)
    : Array3<double>(grid), _origin(grid.GetOrigin())
{

}

ScalarGrid3::~ScalarGrid3()
{

}

double ScalarGrid3::Sample(const Vector3<double>& position) const
{
    return 0;
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

Vector3<double> ScalarGrid3::GetOrigin() const
{
    return _origin;
}

void ScalarGrid3::SetOrigin(Vector3<double> origin)
{
    _origin = origin;
}

