#include "scalar_grid3d.hpp"

ScalarGrid3D::ScalarGrid3D(Vector3<size_t> size, Vector3<double> origin, Vector3<double> spacing, double initialValue): Grid3D(origin, spacing), Array3<double>(size, initialValue) 
{
    
}

ScalarGrid3D::ScalarGrid3D(const ScalarGrid3D& grid) : Grid3D(grid.GetOrigin(), grid.GetGridSpacing()), Array3<double>(grid.GetSize()) 
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