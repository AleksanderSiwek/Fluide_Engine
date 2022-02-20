#include "gird_3d_system.hpp"

Grid3DSystem::Grid3DSystem(Vector3<size_t> size, Vector3<double> dimensions) : _size(size), _dimensions(dimensions)
{

}


Grid3DSystem::~Grid3DSystem()
{

}

void Grid3DSystem::Resize(Vector3<size_t> size, double scalarInitialValue, Vector3<double> vectorInitialValue)
{
    _size = size;

    for (auto& data : _scalarData)
    {
        data.Resize(_size, scalarInitialValue);
    }

    for (auto& data : _vectorData)
    {
        data.Resize(_size, vectorInitialValue);
    }
}

void Grid3DSystem::AddScalarData(std::string name, double initialValue)
{
    _scalarDataDict[name] = _scalarData.size();
    _scalarData.emplace_back(Array3<double>(_size, initialValue));
}

void Grid3DSystem::AddVectorData(std::string name, Vector3<double> initialValue)
{
    _vectorDataDict[name] = _vectorData.size();
    _vectorData.emplace_back(Array3<Vector3<double>>(_size, initialValue));
}

void Grid3DSystem::SetDimensions(Vector3<double> dimensions)
{
    _dimensions = dimensions;
}

void Grid3DSystem::SetScalarValues(size_t idx, double value)
{
    _scalarData[idx].Fill(value);
}

void Grid3DSystem::SetScalarValues(std::string name, double value)
{
    _scalarData[GetScalarIdxByName(name)].Fill(value);
}

void Grid3DSystem::SetScalarValues(size_t idx, Array3<double> value)
{
    _scalarData[idx] = value;
}

void Grid3DSystem::SetScalarValues(std::string name, Array3<double> value)
{
    _scalarData[GetScalarIdxByName(name)] = value;
}

void Grid3DSystem::SetVectorValues(size_t idx, Vector3<double> value)
{
    _vectorData[idx].Fill(value);
}

void Grid3DSystem::SetVectorValues(std::string name, Vector3<double> value)
{
    _vectorData[GetVectorIdxByName(name)].Fill(value);
}

void Grid3DSystem::SetVectorValues(size_t idx, Array3<Vector3<double>> value)
{
    _vectorData[idx] = value;
}

void Grid3DSystem::SetVectorValues(std::string name, Array3<Vector3<double>> value)
{
    _vectorData[GetVectorIdxByName(name)] = value;

}

void Grid3DSystem::SetScalarValue(size_t idx, size_t i, size_t j, size_t k, double value)
{
    _scalarData[idx](i, j, k) = value;
}

void Grid3DSystem::SetScalarValue(std::string name, size_t i, size_t j, size_t k, double value)
{
    _scalarData[GetScalarIdxByName(name)](i, j, k) = value;
}

void Grid3DSystem::SetVectorValue(size_t idx, size_t i, size_t j, size_t k, Vector3<double> value)
{
    _vectorData[idx](i, j, k) = value;
}

void Grid3DSystem::SetVectorValue(std::string name, size_t i, size_t j, size_t k, Vector3<double> value)
{
    _vectorData[GetVectorIdxByName(name)](i, j, k) = value;
}

Vector3<size_t> Grid3DSystem::GetSize() const
{
    return _size;
}

Vector3<double> Grid3DSystem::GetDimensions() const
{
    return _dimensions;
}

Array3<double> Grid3DSystem::GetScalarValues(size_t idx) const
{
    return _scalarData[idx];
}

Array3<double> Grid3DSystem::GetScalarValues(std::string name) const
{
    return _scalarData[GetScalarIdxByName(name)];
}

Array3<Vector3<double>> Grid3DSystem::GetVectorValues(size_t idx) const
{
    return _vectorData[idx];
}

Array3<Vector3<double>> Grid3DSystem::GetVectorValues(std::string name) const
{
    return _vectorData[GetVectorIdxByName(name)];
}

Array3<double>* Grid3DSystem::GetScalarValuesPtr(size_t idx)
{
    return &(_scalarData[idx]);
}

Array3<double>* Grid3DSystem::GetScalarValuesPtr(std::string name)
{
    return &(_scalarData[GetScalarIdxByName(name)]);
}

Array3<Vector3<double>>* Grid3DSystem::GetVectorValuesPtr(size_t idx)
{
    return &(_vectorData[idx]);
}

Array3<Vector3<double>>* Grid3DSystem::GetVectorValuesPtr(std::string name)
{
    return &(_vectorData[GetVectorIdxByName(name)]);
}

double Grid3DSystem::GetScalarValue(size_t idx, size_t i, size_t j, size_t k) const
{
    return _scalarData[idx](i, j, k);
}

double Grid3DSystem::GetScalarValue(std::string name, size_t i, size_t j, size_t k) const
{
    return _scalarData[GetScalarIdxByName(name)](i, j, k);
}

Vector3<double> Grid3DSystem::GetVectorValue(size_t idx, size_t i, size_t j, size_t k) const
{
    return _vectorData[idx](i, j, k);
}

Vector3<double> Grid3DSystem::GetVectorValue(std::string name, size_t i, size_t j, size_t k) const
{
    return _vectorData[GetVectorIdxByName(name)](i, j, k);
}

size_t Grid3DSystem::GetScalarIdxByName(std::string name) const
{
    return _scalarDataDict.at(name);
}

size_t Grid3DSystem::GetVectorIdxByName(std::string name) const
{
    return _vectorDataDict.at(name);
}

size_t Grid3DSystem::GetScalarDataMaxIdx() const
{
    return _scalarData.size();
}

size_t Grid3DSystem::GetVectorDataMaxIdx() const
{
    return _vectorData.size();
}