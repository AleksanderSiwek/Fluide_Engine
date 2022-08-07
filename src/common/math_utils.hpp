#ifndef _MATHS_UTILS_HPP
#define _MATHS_UTILS_HPP

#include <cmath>
#include <array>

#include "vector3.hpp"
#include "array3.hpp"


constexpr double PI = 3.14159265358979323846;

template <typename T>
inline T Clamp(const T& val, const T& maxVal, const T& minVal)
{
    if(val < minVal)
    {
        return minVal;
    }
    else if(val > maxVal)
    {
        return maxVal;
    }
    else
    {
        return val;
    }
}

template <typename S, typename T>
inline S Lerp(const S& a, const S& b, T factor)
{
    return (1 - factor) * a + factor * b;
}

template <typename S, typename T>
inline S Bilerp(const S& x00, const S& x10, const S& x01, const S& x11, T factorX, T factorY)     
{
    return Lerp(Lerp(x00, x10, factorX), Lerp(x01, x11, factorX), factorY);
}  

template <typename S, typename T>
inline S Trilerp(const S& x000, const S& x100, const S& x010, const S& x110, const S& x001, const S& x101, const S& x011, const S& x111, T factorX, T factorY, T factorZ)     
{
    return Lerp(Bilerp(x000, x100, x010, x110, factorX, factorY), Bilerp(x001, x101, x011, x111, factorX, factorY), factorZ);
}  

template<typename T>
inline void GetBarycentric(T x, int iLow, int iHigh, int* i, T* f) {
    T s = std::floor(x);
    *i = static_cast<int>(s);
    int siLow = static_cast<int>(iLow);
    int siHigh = static_cast<int>(iHigh);

    int offset = -siLow;
    siLow += offset;
    siHigh += offset;

    if (siLow == siHigh) {
        *i = siLow;
        *f = 0;
    } else if (*i < siLow) {
        *i = siLow;
        *f = 0;
    } else if (*i > siHigh - 1) {
        *i = siHigh - 1;
        *f = 1;
    } else {
        *f = static_cast<T>(x - s);
    }

    *i -= offset;
}

inline void GetCooridnatesAndWeights(const Vector3<size_t>& size, const Vector3<double>& origin, const Vector3<double>& gridSpacing, 
                              const Vector3<double> x, std::array<Vector3<size_t>, 8>& indexes, std::array<double, 8>& weights)
{
    int i = 0, j = 0, k = 0;
    double fx = 0, fy = 0, fz = 0;

    const int iSize = static_cast<int>(size.x);
    const int jSize = static_cast<int>(size.y);
    const int kSize = static_cast<int>(size.z);

    const Vector3<double> normalizedX = (x - origin) / gridSpacing;

    GetBarycentric<double>(normalizedX.x, 0, iSize - 1, &i, &fx);
    GetBarycentric<double>(normalizedX.y, 0, jSize - 1, &j, &fy);
    GetBarycentric<double>(normalizedX.z, 0, kSize - 1, &k, &fz);

    const int ip1 = std::min(i + 1, iSize - 1);
    const int jp1 = std::min(j + 1, jSize - 1);
    const int kp1 = std::min(k + 1, kSize - 1);

    indexes[0] = Vector3<size_t>(i, j, k);
    indexes[1] = Vector3<size_t>(ip1, j, k);
    indexes[2] = Vector3<size_t>(i, jp1, k);
    indexes[3] = Vector3<size_t>(ip1, jp1, k);
    indexes[4] = Vector3<size_t>(i, j, kp1);
    indexes[5] = Vector3<size_t>(ip1, j, kp1);
    indexes[6] = Vector3<size_t>(i, jp1, kp1);
    indexes[7] = Vector3<size_t>(ip1, jp1, kp1);

    weights[0] = (1.00 - fx) * (1.00 - fy) * (1.00 - fz);
    weights[1] = fx * (1.00 - fy) * (1.00 - fz);
    weights[2] = (1.00 - fx) * fy * (1.00 - fz);
    weights[3] = fx * fy * (1.00 - fz);
    weights[4] = (1.00 - fx) * (1.00 - fy) * fz;
    weights[5] = fx * (1.00 - fy) * fz;
    weights[6] = (1.00 - fx) * fy * fz; 
    weights[7] = fx * fy * fz; 
}

inline void GetCooridnatesAndGradientWeights(const Vector3<size_t>& size, const Vector3<double>& origin, const Vector3<double>& gridSpacing, 
                              const Vector3<double> x, std::array<Vector3<size_t>, 8>& indexes, std::array<Vector3<double>, 8>& weights)
{
    int i = 0, j = 0, k = 0;
    double fx = 0, fy = 0, fz = 0;

    const int iSize = static_cast<int>(size.x);
    const int jSize = static_cast<int>(size.y);
    const int kSize = static_cast<int>(size.z);

    const Vector3<double> normalizedX = (x - origin) / gridSpacing;
    const Vector3<double> invGridSpacing = Vector3<double>(1, 1, 1) / gridSpacing;

    GetBarycentric<double>(normalizedX.x, 0, iSize - 1, &i, &fx);
    GetBarycentric<double>(normalizedX.y, 0, jSize - 1, &j, &fy);
    GetBarycentric<double>(normalizedX.z, 0, kSize - 1, &k, &fz);

    const int ip1 = std::min(i + 1, iSize - 1);
    const int jp1 = std::min(j + 1, jSize - 1);
    const int kp1 = std::min(k + 1, kSize - 1);

    indexes[0] = Vector3<size_t>(i, j, k);
    indexes[1] = Vector3<size_t>(ip1, j, k);
    indexes[2] = Vector3<size_t>(i, jp1, k);
    indexes[3] = Vector3<size_t>(ip1, jp1, k);
    indexes[4] = Vector3<size_t>(i, j, kp1);
    indexes[5] = Vector3<size_t>(ip1, j, kp1);
    indexes[6] = Vector3<size_t>(i, jp1, kp1);
    indexes[7] = Vector3<size_t>(ip1, jp1, kp1);

    weights[0] = Vector3<double>(
        -invGridSpacing.x * (1 - fy) * (1 - fz),
        -invGridSpacing.y * (1 - fx) * (1 - fz),
        -invGridSpacing.z * (1 - fx) * (1 - fy));
    weights[1] = Vector3<double>(
        invGridSpacing.x * (1 - fy) * (1 - fz),
        fx * (-invGridSpacing.y) * (1 - fz),
        fx * (1 - fy) * (-invGridSpacing.z));
    weights[2] = Vector3<double>(
        (-invGridSpacing.x) * fy * (1 - fz),
        (1 - fx) * invGridSpacing.y * (1 - fz),
        (1 - fx) * fy * (-invGridSpacing.z));
    weights[3] = Vector3<double>(
        invGridSpacing.x * fy * (1 - fz),
        fx * invGridSpacing.y * (1 - fz),
        fx * fy * (-invGridSpacing.z));
    weights[4] = Vector3<double>(
        (-invGridSpacing.x) * (1 - fy) * fz,
        (1 - fx) * (-invGridSpacing.y) * fz,
        (1 - fx) * (1 - fy) * invGridSpacing.z);
    weights[5] = Vector3<double>(
        invGridSpacing.x * (1 - fy) * fz,
        fx * (-invGridSpacing.y) * fz,
        fx * (1 - fy) * invGridSpacing.z);
    weights[6] = Vector3<double>(
        (-invGridSpacing.x) * fy * fz,
        (1 - fx) * invGridSpacing.y * fz,
        (1 - fx) * fy * invGridSpacing.z);
    weights[7] = Vector3<double>(
        invGridSpacing.x * fy * fz,
        fx * invGridSpacing.y * fz,
        fx * fy * invGridSpacing.z);
}

template<typename T>
inline Vector3<T> Cross(const Vector3<T>& a, const Vector3<T>& b)
{
    return Vector3<T>(a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

inline void ExtrapolateToRegion(const Array3<double>& input, const Array3<int>& valid, size_t numberOfIterations, Array3<double>& output)
{
    const Vector3<size_t>& size = input.GetSize();
    Array3<int> valid0(valid);
    Array3<int> valid1(valid);
    
    valid0.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        valid0(i, j, k) = valid(i, j, k);
        output(i, j, k) = input(i, j, k);
    });

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter)
    {
        valid0.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
        {
            double sum = 0;
            unsigned int count = 0;

            if (!valid0(i, j, k)) 
            {
                if (i + 1 < size.x && valid0(i + 1, j, k)) 
                {
                    sum += output(i + 1, j, k);
                    ++count;
                }
                if (i > 0 && valid0(i - 1, j, k)) 
                {
                    sum += output(i - 1, j, k);
                    ++count;
                }

                if (j + 1 < size.y && valid0(i, j + 1, k)) 
                {
                    sum += output(i, j + 1, k);
                    ++count;
                }
                if (j > 0 && valid0(i, j - 1, k)) 
                {
                    sum += output(i, j - 1, k);
                    ++count;
                }

                if (k + 1 < size.z && valid0(i, j, k + 1)) 
                {
                    sum += output(i, j, k + 1);
                    ++count;
                }
                if (k > 0 && valid0(i, j, k - 1)) 
                {
                    sum += output(i, j, k - 1);
                    ++count;
                }

                if (count > 0) 
                {
                    output(i, j, k)= sum / count;
                    valid1(i, j, k) = 1;
                }
            } 
            else 
            {
                valid1(i, j, k) = 1;
            }
        });
        valid0.Swap(valid1);
    }
}

#endif // _MATHS_UTILS_HPP