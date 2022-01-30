#ifndef VECTOR_FIELD_3_HPP
#define VECTOR_FIELD_3_HPP

#include "vector3.hpp"


template <typename T>
class VectorField3
{
    public:
        VectorField3() {}

        virtual ~VectorField3() {}

        virtual Vector3<T> GetSample(const Vector3<int>& position) const = 0;

        virtual Vector3<T> Divergence(const Vector3<int>& position) const = 0;

        virtual Vector3<T> Curl(const Vector3<int>& position) const = 0;
};

#endif // VECTOR_FIELD_3_HPP