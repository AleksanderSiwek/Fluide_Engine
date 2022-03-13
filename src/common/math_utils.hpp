#ifndef _MATHS_UTILS_HPP
#define _MATHS_UTILS_HPP

#include <cmath>
#include "vector3.hpp"


template <typename T>
inline T Clamp(const T& val, const T& maxVal, const T& minVal)
{
    return std::max(minVal, std::min(val, maxVal));
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
inline void GetBarycentric(T x, size_t iLow, size_t iHigh, size_t* i, T* f) {
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

#endif // _MATHS_UTILS_HPP