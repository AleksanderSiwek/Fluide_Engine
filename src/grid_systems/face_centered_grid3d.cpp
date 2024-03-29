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

std::vector<double> FaceCenteredGrid3D::Serialize() const
{
    std::vector<double> serialized;
    // size_t size = _dataX.GetRawData().size() + _dataY.GetRawData().size() + _dataZ.GetRawData().size();
    // serialized.reserve(size);
    // serialized.insert(serialized.end(), _dataX.GetRawData().begin(), _dataX.GetRawData().end());
    // serialized.insert(serialized.end(), _dataY.GetRawData().begin(), _dataY.GetRawData().end());
    // serialized.insert(serialized.end(), _dataZ.GetRawData().begin(), _dataZ.GetRawData().end());
    return serialized;
}


Vector3<size_t> FaceCenteredGrid3D::GetSize() const
{
    return _size; //- Vector3<size_t>(1, 1, 1)); // TO DO
}

Vector3<size_t> FaceCenteredGrid3D::GetXSize() const
{
    return _dataX.GetSize();
}

Vector3<size_t> FaceCenteredGrid3D::GetYSize() const
{
    return _dataY.GetSize();
}

Vector3<size_t> FaceCenteredGrid3D::GetZSize() const
{
    return _dataZ.GetSize();
}

Vector3<size_t> FaceCenteredGrid3D::GetActualSize() const
{
    return _size;
}

Vector3<double> FaceCenteredGrid3D::GetDiemensions() const
{
    return Vector3<double>(_size.x * _gridSpacing.x, _size.y * _gridSpacing.y, _size.z * _gridSpacing.z);
}


void FaceCenteredGrid3D::Resize(Vector3<size_t> size)
{
    SetSize(size);
    _dataX.Resize(_size + Vector3<size_t>(1, 0, 0));
    _dataY.Resize(_size + Vector3<size_t>(0, 1, 0));
    _dataZ.Resize(_size + Vector3<size_t>(0, 0, 1));
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

void FaceCenteredGrid3D::ParallelFill(double xVal, double yVal, double zVal)
{
    _dataX.ParallelFill(xVal);
    _dataY.ParallelFill(yVal);
    _dataZ.ParallelFill(zVal);
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

Vector3<double> FaceCenteredGrid3D::GetElement(size_t i, size_t j, size_t k) const
{
    return Vector3<double>(x(i, j, k), y(i, j, k), z(i, j, k));
}

Vector3<double> FaceCenteredGrid3D::Sample(const Vector3<double>& position) const
{
    double x = SampleArray(position, _dataXOrigin, _dataX);
    double y = SampleArray(position, _dataYOrigin, _dataY);
    double z = SampleArray(position, _dataZOrigin, _dataZ);
    return Vector3<double>(x, y, z);
}

Vector3<double> FaceCenteredGrid3D::ValueAtCellCenter(size_t i, size_t j, size_t k) const
{
    double left = _dataX(i, j, k);
    double right = _dataX(i + 1, j, k);
    double down = _dataY(i, j ,k);
    double up = _dataY(i, j + 1, k);
    double back = _dataZ(i, j, k);
    double front = _dataZ(i, j, k + 1);
    return 0.5 * Vector3<double>(left + right, down + up, back + front);
}

double FaceCenteredGrid3D::DivergenceAtCallCenter(size_t i, size_t j, size_t k) const
{
    double left = _dataX(i, j, k);
    double right = _dataX(i + 1, j, k);
    double down = _dataY(i, j ,k);
    double up = _dataY(i, j + 1, k);
    double back = _dataZ(i, j, k);
    double front = _dataZ(i, j, k + 1);
    return (right - left)/_gridSpacing.x + (up - down)/_gridSpacing.y + (front - back)/_gridSpacing.z;
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

void FaceCenteredGrid3D::ForEachIndex(std::function<void(size_t, size_t, size_t)>& functor)
{
    for(size_t i = 0; i < _size.x; i++)
    {
        for(size_t j = 0; j < _size.y; j++)
        {
            for(size_t k = 0; k < _size.z; k++)
            {
                functor(i, j, k);
            }
        }
    }
}

void FaceCenteredGrid3D::ForEachIndex(const std::function<void(size_t, size_t, size_t)>& functor) const
{
    for(size_t i = 0; i < _size.x; i++)
    {
        for(size_t j = 0; j < _size.y; j++)
        {
            for(size_t k = 0; k < _size.z; k++)
            {
                functor(i, j, k);
            }
        }
    }
}

void FaceCenteredGrid3D::ParallelForEachIndex(std::function<void(size_t, size_t, size_t)>& functor)
{
    parallel_utils::ForEach3(_size.x, _size.y, _size.z, functor);
}

void FaceCenteredGrid3D::ParallelForEachIndex(const std::function<void(size_t, size_t, size_t)>& functor) const
{
    parallel_utils::ConstForEach3(_size.x, _size.y, _size.z, functor);
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

Array3<double>& FaceCenteredGrid3D::GetDataXRef()
{
    return _dataX;
}

Array3<double>& FaceCenteredGrid3D::GetDataYRef()
{
    return _dataY;
}

Array3<double>& FaceCenteredGrid3D::GetDataZRef()
{
    return _dataZ;
}

Vector3<double> FaceCenteredGrid3D::GetXPos(size_t i, size_t j, size_t k) const
{
    return _dataXOrigin + _gridSpacing * Vector3<double>((double)i, (double)j, (double)k);
}

Vector3<double> FaceCenteredGrid3D::GetYPos(size_t i, size_t j, size_t k) const
{
    return _dataYOrigin + _gridSpacing * Vector3<double>((double)i, (double)j, (double)k);
}

Vector3<double> FaceCenteredGrid3D::GetZPos(size_t i, size_t j, size_t k) const
{
    return _dataZOrigin + _gridSpacing * Vector3<double>((double)i, (double)j, (double)k);
}

Vector3<double> FaceCenteredGrid3D::GetCellCenterPos(size_t i, size_t j, size_t k) const
{
    return _origin + 0.5 * _gridSpacing + Vector3<double>(_gridSpacing.x * i, _gridSpacing.y * j, _gridSpacing.z * k);
}


void FaceCenteredGrid3D::CalculateDataOrigins()
{
    // _dataXOrigin = GetOrigin() + GetGridSpacing() * Vector3<double>(0, 0.5, 0.5);
    // _dataYOrigin = GetOrigin() + GetGridSpacing() * Vector3<double>(0.5, 0, 0.5);
    // _dataZOrigin = GetOrigin() + GetGridSpacing() * Vector3<double>(0.5, 0.5, 0);
    _dataXOrigin = _origin + _gridSpacing * Vector3<double>(0, 0.5, 0.5);
    _dataYOrigin = _origin + _gridSpacing * Vector3<double>(0.5, 0, 0.5);
    _dataZOrigin = _origin + _gridSpacing * Vector3<double>(0.5, 0.5, 0);
}

void FaceCenteredGrid3D::SetSize(Vector3<size_t> size)
{
    _size = size; // TO DO + Vector3<size_t>(1);
}

double FaceCenteredGrid3D::SampleArray(const Vector3<double>& position, const Vector3<double>& origin, const Array3<double>& arr) const
{
    int i, j, k;
    double factorX, factorY, factorZ ;
    i = j = k = 0;
    factorX = factorY = factorZ = 0;

    Vector3<double> normalizedPoistion = (position - origin) / _gridSpacing;
    const auto& size = arr.GetSize();
    int sizeX = static_cast<int>(size.x);
    int sizeY = static_cast<int>(size.y);
    int sizeZ = static_cast<int>(size.z);

    GetBarycentric<double>(normalizedPoistion.x, 0, sizeX - 1, &i, &factorX);
    GetBarycentric<double>(normalizedPoistion.y, 0, sizeY - 1, &j, &factorY);
    GetBarycentric<double>(normalizedPoistion.z, 0, sizeZ - 1, &k, &factorZ);

    size_t ip1 = std::min(i + 1, sizeX - 1);
    size_t jp1 = std::min(j + 1, sizeY - 1);
    size_t kp1 = std::min(k + 1, sizeZ - 1);

    return Trilerp<double, double>(arr(i, j, k),
                                            arr(ip1, j, k),
                                            arr(i, jp1, k),
                                            arr(ip1, jp1, k),
                                            arr(i, j, kp1),
                                            arr(ip1, j, kp1),
                                            arr(i, jp1, kp1),
                                            arr(ip1, jp1, kp1),
                                            factorX,
                                            factorY,
                                            factorZ);
}
