#ifndef GRID_3D_SYSTEM_HPP
#define GRID_3D_SYSTEM_HPP

#include <vector>
#include <map>
#include <string>
#include "../common/vector3.hpp"
#include "../common/array3.hpp"

class Grid3DSystem
{
    public:
        Grid3DSystem(Vector3<size_t> size=1, Vector3<double> dimensions=1);
        
        ~Grid3DSystem();

        void Resize(Vector3<size_t> size, double scalarInitialValue=0, Vector3<double> vectorInitialValue=(0, 0, 0));
        void AddScalarData(std::string name, double initialValue=0);
        void AddVectorData(std::string name, Vector3<double> initialValue=(0, 0, 0));

        void SetDimensions(Vector3<double> dimensions);
        void SetScalarValues(size_t idx, double value);
        void SetScalarValues(std::string name, double value);
        void SetScalarValues(size_t idx, Array3<double> value);
        void SetScalarValues(std::string name, Array3<double> value);
        void SetVectorValues(size_t idx, Vector3<double> value);
        void SetVectorValues(std::string name, Vector3<double> value);
        void SetVectorValues(size_t idx, Array3<Vector3<double>> value);
        void SetVectorValues(std::string name, Array3<Vector3<double>> value);
        void SetScalarValue(size_t idx, size_t i, size_t j, size_t k, double value);
        void SetScalarValue(std::string name, size_t i, size_t j, size_t k, double value);
        void SetVectorValue(size_t idx, size_t i, size_t j, size_t k, Vector3<double> value);
        void SetVectorValue(std::string name, size_t i, size_t j, size_t k, Vector3<double> value);

        Vector3<size_t> GetSize() const;
        Vector3<double> GetDimensions() const;
        Array3<double> GetScalarValues(size_t idx) const;
        Array3<double> GetScalarValues(std::string name) const;
        Array3<Vector3<double>> GetVectorValues(size_t idx) const;
        Array3<Vector3<double>> GetVectorValues(std::string name) const;
        Array3<double>* GetScalarValuesPtr(size_t idx);
        Array3<double>* GetScalarValuesPtr(std::string name);
        Array3<Vector3<double>>* GetVectorValuesPtr(size_t idx);
        Array3<Vector3<double>>* GetVectorValuesPtr(std::string name);
        double GetScalarValue(size_t idx, size_t i, size_t j, size_t k) const;
        double GetScalarValue(std::string name, size_t i, size_t j, size_t k) const;
        Vector3<double> GetVectorValue(size_t idx, size_t i, size_t j, size_t k) const;
        Vector3<double> GetVectorValue(std::string name, size_t i, size_t j, size_t k) const;
        size_t GetScalarIdxByName(std::string name) const;
        size_t GetVectorIdxByName(std::string name) const;
        size_t GetScalarDataMaxIdx() const;
        size_t GetVectorDataMaxIdx() const;

    private:
        Vector3<size_t> _size;
        Vector3<double> _dimensions;

        std::map<std::string, size_t> _scalarDataDict;
        std::map<std::string, size_t> _vectorDataDict;
        std::vector<Array3<double>> _scalarData;
        std::vector<Array3<Vector3<double>>> _vectorData;

};

#endif // GRID_3D_SYSTEM_HPP