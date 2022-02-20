#include "scalar_grid3d.hpp"

ScalarGrid3D::ScalarGrid3D(Vector3<size_t> size, Vector3<double> origin, Vector3<double> spacing, double initialValue): Grid3D(size, origin, spacing, initialValue) 
{
    
}

ScalarGrid3D::ScalarGrid3D(const ScalarGrid3D& grid) : Grid3D<double>(grid.GetSize(), grid.GetOrigin(), grid.GetGridSpacing())
{
    Fill(grid.GetRawData());
}

ScalarGrid3D::~ScalarGrid3D()
{

}


Vector3<double> ScalarGrid3D::Gradient(const Vector3<double>& pos) 
{ 
    // TO DO
    return 0;
}

double ScalarGrid3D::Laplacian(const Vector3<double>& pos) 
{ 
    // TO DO
    return 0;
}