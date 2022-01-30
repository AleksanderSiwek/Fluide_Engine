#ifndef ARRAY_3_HPP
#define ARRAY_3_HPP

#include <vector>
#include "vector3.hpp"


template <typename T>
class Array3
{
    public:
        Array3() : _size(Vector3<size_t>(0, 0, 0))
        {

        }

        Array3(size_t width, size_t height, size_t depth, const T& initailValue = T()) : _size(Vector3<size_t>(width, height, depth)), _data(std::vector<T>(width * height * depth, initailValue))
        {

        }

        Array3(const Vector3<size_t>& size, const T& initailValue = T()) : _size(size), _data(std::vector<T>(size.x * size.y * size.z, initailValue)) 
        {

        }

        Array3(const Array3<T>& array) : _size(array.GetSize()), _data(array.GetRawData()) 
        {

        }

        ~Array3() {}

        Vector3<size_t> GetSize() const
        {
             return _size;
        }

        std::vector<T> GetRawData() const
        {
            return _data;
        }

        const T& GetElement(size_t i, size_t j, size_t k) const
        {      
            
            return _data[i + _size.x * (j + _size.y * k)];
        }
       
        T& GetElement(size_t i, size_t j, size_t k)
        {      
            return _data.at(i + _size.x * (j + _size.y * k));
        }

        const T& GetElement(const Vector3<size_t>& pos) const
        {      
            
            return _data[pos.x + _size.x * (pos.y + _size.y * pos.z)];
        }
       
        T& GetElement(const Vector3<size_t>& pos)
        {      
            return _data.at(pos.x + _size.x * (pos.y + _size.y * pos.z));
        }

        void SetElement(size_t i, size_t j, size_t k, T value)
        {
            _data[i + _size.x * (j + _size.y * k)] = value;
        }

        bool IsEqual(const Array3<T>& arr)
        {
            if(_data.size() != arr.GetRawData().size()) return false;
            if(_size != arr.GetSize()) return false;

            for(size_t i = 0; i < _data.size(); i++)
            {
                if(_data[i] != arr[i]) return false;
            }

            return true;
        }

        void Fill(T value)
        {
            for(size_t i = 0; i < _data.size(); i++)
            {
                _data[i] = value;
            }
        }

        void Fill(const Array3& array)
        {
            std::vector<T> data = array.GetRawData();

            for(size_t i = 0; i < _data.size(); i++)
            {
                if(i < data.size())
                    _data[i] = data[i];
                else
                    _data[i] = T();
            }
        }

        void Fill(const std::vector<T>& data)
        {
            for(size_t i = 0; i < _data.size(); i++)
            {
                if(i < data.size())
                    _data[i] = data[i];
                else
                    _data[i] = T();
            }
        }

        void Resize(const Vector3<size_t> size)
        {
            std::vector<T> data = _data;
            Clear();
            _size = size;
            _data = std::vector<T>(_size.x * _size.y * _size.z, 0);

            Fill(data);
        }

        void Resize(size_t x_size, size_t y_size, size_t z_size)
        {
            std::vector<T> data = _data;
            Clear();
            _size = Vector3<size_t>(x_size, y_size, z_size);
            _data = std::vector<T>(_size.x * _size.y * _size.z, 0);

            Fill(data);
        }

        void Copy(const Array3& array)
        {
            Clear();
            _size = array.GetSize();
            _data = array.GetRawData();
        }

        void Clear()
        {
            _data.clear();
            _size = Vector3<size_t>(0, 0, 0);
        }

        bool operator==(const Array3<T>& array)
        {
            return IsEqual(array);
        }

        bool operator!=(const Array3<T>& array)
        {
            return !IsEqual(array);
        }

        Array3<T> operator=(const Array3<T>& array)
        {
            Copy(array);
            return *this;
        }

        T& operator()(size_t i, size_t j, size_t k)
        {
            return GetElement(i, j, k);
        }

        const T& operator()(size_t i, size_t j, size_t k) const
        {
            return GetElement(i, j, k);
        }

        T& operator()(const Vector3<size_t>& pos)
        {
            return GetElement(pos);
        }

        const T& operator()(const Vector3<size_t>& pos) const
        {
            return GetElement(pos);
        }

        const T& operator[](size_t i) const
        {
            return _data[i];
        }

        T& operator[](size_t i) 
        {
            return _data[i];
        }

    private:
        std::vector<T> _data;
        Vector3<size_t> _size;
};

#endif // ARRAY_3_HPP