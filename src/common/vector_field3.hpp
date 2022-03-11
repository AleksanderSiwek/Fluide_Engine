#ifndef VECTOR_FIELD_3_HPP
#define VECTOR_FIELD_3_HPP

#include "vector3.hpp"


class VectorField3
{
    public:
        VectorField3() {}

        virtual ~VectorField3() {}

        virtual Vector3<double> Sample(const Vector3<double>& position) const = 0;

        virtual Vector3<double> Divergence(const Vector3<double>& position) const = 0;

        virtual Vector3<double> Curl(const Vector3<double>& position) const = 0;
};

#endif // VECTOR_FIELD_3_HPP