#include "face_centered_grid3d.hpp"

FaceCenteredGrid3D::FaceCenteredGrid3D(Vector3<size_t> size, Vector3<double> origin, Vector3<double> spacing, Vector3<double> initalValue)
    : Grid3D(origin, spacing)
{
    Resize(size);
    Fill(initalValue.x, initalValue.y, initalValue.z);  
    CalculateDataOrigins();
}

FaceCenteredGrid3D::FaceCenteredGrid3D(const FaceCenteredGrid3D& grid) 
    : Grid3D(grid.GetOrigin(), grid.GetGridSpacing())
{   
    Resize(grid.GetSize());
    Fill(grid.GetDataX(), grid.GetDataY(), grid.GetDataZ());  
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

void FaceCenteredGrid3D::Resize(Vector3<size_t> size)
{
    SetSize(size);
    _dataX.Resize(_size);
    _dataY.Resize(_size);
    _dataZ.Resize(_size);
}

void FaceCenteredGrid3D::Fill(double xVal, double yVal, double zVal)
{
    _dataX.Fill(xVal);
    _dataY.Fill(yVal);
    _dataZ.Fill(zVal);
}

void FaceCenteredGrid3D::Fill(const Array3<double>& dataX, const Array3<double>& dataY, const Array3<double>& dataZ)
{
    _dataX.Fill(dataX);
    _dataY.Fill(dataY);
    _dataZ.Fill(dataZ);
}

bool FaceCenteredGrid3D::IsEqual(const FaceCenteredGrid3D& grid)
{
    if(_dataX.IsEqual(grid.GetDataX()) && _dataY.IsEqual(grid.GetDataY()) && _dataZ.IsEqual(grid.GetDataZ()))
        return true;
    return false;
}

double& FaceCenteredGrid3D::x(size_t i, size_t j, size_t k)
{
    return _dataX(i, j, k);
}

const double& FaceCenteredGrid3D::x(size_t i, size_t j, size_t k) const
{
    return _dataX(i, j, k);
}

double& FaceCenteredGrid3D::y(size_t i, size_t j, size_t k)
{
    return _dataY(i, j, k);
}

const double& FaceCenteredGrid3D::y(size_t i, size_t j, size_t k) const
{
    return _dataY(i, j, k);
}

double& FaceCenteredGrid3D::z(size_t i, size_t j, size_t k)
{
    return _dataZ(i, j, k);
}

const double& FaceCenteredGrid3D::z(size_t i, size_t j, size_t k) const
{
    return _dataZ(i, j, k);
}

Vector3<double> FaceCenteredGrid3D::ValueAtCellCenter(size_t i, size_t j, size_t k) const
{
    return 0.5 * Vector3<double>(x(i, j, k) + x(i+1, j, k), y(i, j, k) + y(i, j+1, k), z(i, j, k) + z(i, j, k+1));
}

double FaceCenteredGrid3D:: DivergenceAtCallCenter(size_t i, size_t j, size_t k) const
{
    return (x(i, j, k) - x(i+1, j, k))/_gridSpacing.x + (y(i, j, k) - y(i, j+1, k))/_gridSpacing.y + (z(i, j, k) - z(i, j, k+1))/_gridSpacing.z;
}

Vector3<double> FaceCenteredGrid3D::CurlAtCellCentre(size_t i, size_t j, size_t k) const
{
    Vector3<double> left = ValueAtCellCenter((i > 0) ? i - 1 : i, j, k);
    Vector3<double> right = ValueAtCellCenter((i + 1 < _size.x) ? i + 1 : i, j, k);
    Vector3<double> up = ValueAtCellCenter(i, (j > 0) ? j - 1 : j, k);
    Vector3<double> down = ValueAtCellCenter(i, (j + 1 < _size.y) ? j + 1 : j, k);
    Vector3<double> front = ValueAtCellCenter(i, j, (k > 0) ? k - 1 : k);
    Vector3<double> back = ValueAtCellCenter(i, j, (k + 1 < _size.z) ? k + 1 : k);

    double x_val = 0.5 * ((up.z - down.z) / _gridSpacing.y - (front.y - back.y) / _gridSpacing.z);
    double y_val = 0.5 * ((front.x - back.x) / _gridSpacing.z - (right.z - left.z) / _gridSpacing.x);
    double z_val = 0.5 * ((right.y - left.y) / _gridSpacing.x - (up.x - down.x) / _gridSpacing.y);

    return Vector3<double>(x_val, y_val, z_val);
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

Array3<double> FaceCenteredGrid3D::GetDataX() const
{
    return _dataX;
}

Array3<double> FaceCenteredGrid3D::GetDataY() const
{
    return _dataY;
}

Array3<double> FaceCenteredGrid3D::GetDataZ() const
{
    return _dataZ;
}

Array3<double>* FaceCenteredGrid3D::GetDataXPtr()
{
    return &_dataX;
}

Array3<double>* FaceCenteredGrid3D::GetDataYPtr()
{
    return &_dataY;
}

Array3<double>* FaceCenteredGrid3D::GetDataZPtr()
{
    return &_dataZ;
}

const Array3<double>& FaceCenteredGrid3D::GetDataXRef() const 
{
    return _dataX;
}

const Array3<double>& FaceCenteredGrid3D::GetDataYRef() const
{
    return _dataY;
}   

const Array3<double>& FaceCenteredGrid3D::GetDataZRef() const
{
    return _dataZ;
}

void FaceCenteredGrid3D::CalculateDataOrigins()
{
    _dataXOrigin = GetOrigin() + 0.5 * Vector3<double>(0, GetGridSpacing().y, GetGridSpacing().z);
    _dataYOrigin = GetOrigin() + 0.5 * Vector3<double>(GetGridSpacing().x, 0, GetGridSpacing().z);
    _dataZOrigin = GetOrigin() + 0.5 * Vector3<double>(GetGridSpacing().x, GetGridSpacing().y, 0);
}

void FaceCenteredGrid3D::SetSize(Vector3<size_t> size)
{
    _size = size + Vector3<size_t>(1);
}