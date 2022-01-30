#ifndef SCALAR_FIELD_3_HPP
#define SCALAR_FIELD_3_HPP

#include "vector3.hpp"


template <typename T>
class ScalarField3
{
    public:
        ScalarField3() {}

        virtual ~ScalarField3() {}

        virtual T GetSample(const Vector3<int>& position) const = 0;

        virtual Vector3<T> Gradient(const Vector3<int>& position) const = 0;

        virtual T Laplacian(const Vector3<int>& position) const = 0;
};

#endif // SCALAR_FIELD_3_HPP