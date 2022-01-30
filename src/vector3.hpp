#ifndef VECTOR_3_HPP
#define VECTOR_3_HPP

#include "frame.hpp"

template <typename T>
class Vector3
{
    public:
        T x;
        T y;
        T z;

        Vector3() : x(0), y(0), z(0) {}

        Vector3(T val) : x(val), y(val), z(val) {}

        Vector3(T x_val, T y_val, T z_val) : x(x_val), y(y_val), z(z_val) {}

        Vector3(const Vector3<T>& v) : x(v.x), y(v.y), z(v.z) {}

        ~Vector3() {}

        Vector3<T> Add(const Vector3<T>& v)
        {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        Vector3<T> Add(T val)
        {
            x += val;
            y += val;
            z += val;
            return *this;
        }

        Vector3<T> Subtract(const Vector3<T>& v)
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        Vector3<T> Subtract(T val)
        {
            x -= val;
            y -= val;
            z -= val;
            return *this;
        }

        Vector3<T> Multiply(T val)
        {
            x *= val;
            y *= val;
            z *= val;
            return *this;
        }

        Vector3<T> Multiply(const Vector3<T>& v)
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }

        Vector3<T> Divide(T val)
        {
            x /= val;
            y /= val;
            z /= val;
            return *this;
        }

        Vector3<T> Divide(const Vector3<T>& v)
        {
            x /= v.x;
            y /= v.y;
            z /= v.z;
            return *this;
        }

        T Max() 
        {
            if(x >= y && x >= z) return x;
            if(y >= x && y >= z) return y;
            return z;
        }

        T Min()
        {
            if(x <= y && x <= z) return x;
            if(y <= x && y <= z) return y;
            return z;
        }

        void Normalize()
        {
            Divide(sqrt(x * x + y * y + z * z));
        }

        bool IsEqual(const Vector3<T>& v) const
        {
            return x == v.x && y == v.y && z == v.z;
        }

        bool IsEqual(T val) const
        {
            return x == val && y == val && z == val;
        }

        Vector3<T> operator=(const Vector3<T>& v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
            return *this;
        }

        Vector3<T> operator=(T val)
        {
            x = val;
            y = val;
            z = val;
            return *this;
        }

        Vector3<T> operator+=(T val)
        {
            return Add(val);
        }

        Vector3<T> operator+=(const Vector3<T>& v)
        {
            return Add(v);
        }

        Vector3<T> operator-=(T val)
        {
            return Subtract(val);
        }

        Vector3<T> operator-=(const Vector3<T>& v)
        {
            return Subtract(v);
        }

        Vector3<T> operator*=(T val)
        {
            return Multiply(val);
        }

        Vector3<T> operator*=(const Vector3<T>& v)
        {
            return Multiply(v);
        }

        Vector3<T> operator/=(T val)
        {
            return Divide(val);
        }

        Vector3<T> operator/=(const Vector3<T>& v)
        {
            return Divide(v);
        }

        bool operator==(const Vector3<T>& v) const
        {
            return IsEqual(v);
        }

        bool operator==(T val) const
        {
            return IsEqual(val);
        }

        bool operator!=(const Vector3<T>& v) const
        {
            return !IsEqual(v);
        }

        bool operator!=(T val) const
        {
            return !IsEqual(val);
        }
};

template <typename T>
Vector3<T> operator+(T val, const Vector3<T>& v)
{
    return Vector3<T>(v).Add(val);
}

template <typename T>
Vector3<T> operator+(const Vector3<T>& v, T val)
{
    return Vector3<T>(v).Add(val);
}

template <typename T>
Vector3<T> operator+(const Vector3<T>& v1, const Vector3<T>& v2)
{
    return Vector3<T>(v1).Add(v2);
}

template <typename T>
Vector3<T> operator-(T val, const Vector3<T>& v)
{
    return Vector3<T>(val).Subtract(v);
}

template <typename T>
Vector3<T> operator-(const Vector3<T>& v, T val)
{
    return Vector3<T>(v).Subtract(val);
}

template <typename T>
Vector3<T> operator-(const Vector3<T>& v1, const Vector3<T>& v2)
{
    return Vector3<T>(v1).Subtract(v2);
}

template <typename T>
Vector3<T> operator*(T val, const Vector3<T>& v)
{
    return Vector3<T>(val).Multiply(v);
}

template <typename T>
Vector3<T> operator*(const Vector3<T>& v, T val)
{
    return Vector3<T>(v).Multiply(val);
}

template <typename T>
Vector3<T> operator*(const Vector3<T>& v1, const Vector3<T>& v2)
{
    return Vector3<T>(v1).Multiply(v2);
}

template <typename T>
Vector3<T> operator/(T val, const Vector3<T>& v)
{
    return Vector3<T>(val).Divide(v);
}

template <typename T>
Vector3<T> operator/(const Vector3<T>& v, T val)
{
    return Vector3<T>(v).Divide(val);
}

template <typename T>
Vector3<T> operator/(const Vector3<T>& v1, const Vector3<T>& v2)
{
    return Vector3<T>(v1).Divide(v2);
}

#endif // VECTOR_3_HPP