#include "face_centered_grid3d.hpp"

FaceCenteredGrid3D::FaceCenteredGrid3D(Vector3<size_t> size, Vector3<double> origin, Vector3<double> spacing, Vector3<double> initialValue) 
    : Grid3D(size, origin, spacing, initialValue)
{
    Resize(size);
    Fill(initialValue);
    CalculateDataOrigins();
}

FaceCenteredGrid3D::FaceCenteredGrid3D(const FaceCenteredGrid3D& grid) 
    : Grid3D<Vector3<double>>(grid.GetSize(), grid.GetOrigin(), grid.GetGridSpacing())
{   
    Resize(grid.GetSize());
    Fill(grid.GetRawData());  
    CalculateDataOrigins();
}

FaceCenteredGrid3D::~FaceCenteredGrid3D()
{

}

Vector3<size_t> FaceCenteredGrid3D::GetSize() const
{
    return (_size - Vector3<size_t>(1, 1, 1));
}

Vector3<size_t> FaceCenteredGrid3D::GetActualSize() const
{
    return _size;
}

double& FaceCenteredGrid3D::x(size_t i, size_t j, size_t k)
{
    return GetElement(i, j, k).x;
}

const double& FaceCenteredGrid3D::x(size_t i, size_t j, size_t k) const
{
    return GetElement(i, j, k).x;
}

double& FaceCenteredGrid3D::y(size_t i, size_t j, size_t k)
{
    return GetElement(i, j, k).y;
}

const double& FaceCenteredGrid3D::y(size_t i, size_t j, size_t k) const
{
    return GetElement(i, j, k).y;
}

double& FaceCenteredGrid3D::z(size_t i, size_t j, size_t k)
{
    return GetElement(i, j, k).z;
}

const double& FaceCenteredGrid3D::z(size_t i, size_t j, size_t k) const
{
    return GetElement(i, j, k).z;
}

void FaceCenteredGrid3D::SetGridSpacing(Vector3<double> gridSpacing)
{
    _gridSpacing = gridSpacing;
    CalculateDataOrigins();
}

Vector3<double> FaceCenteredGrid3D::GetDataXOrigin() const
{
    return _dataXOrigin;
}

Vector3<double> FaceCenteredGrid3D::GetDataYOrigin() const
{
    return _dataYOrigin;
}

Vector3<double> FaceCenteredGrid3D::GetDataZOrigin() const
{
    return _dataZOrigin;
}

void FaceCenteredGrid3D::CalculateDataOrigins()
{
    _dataXOrigin = GetOrigin() + 0.5 * Vector3<double>(0, GetGridSpacing().y, GetGridSpacing().z);
    _dataYOrigin = GetOrigin() + 0.5 * Vector3<double>(GetGridSpacing().x, 0, GetGridSpacing().z);
    _dataZOrigin = GetOrigin() + 0.5 * Vector3<double>(GetGridSpacing().x, GetGridSpacing().y, 0);
}

void FaceCenteredGrid3D::SetSize(Vector3<size_t> size)
{
    _size = size + Vector3<size_t>(1, 1, 1);
}