#ifndef SCALAR_FIELD_3_HPP
#define SCALAR_FIELD_3_HPP

#include "vector3.hpp"


class ScalarField3
{
    public:
        ScalarField3() {}

        virtual ~ScalarField3() {}

        virtual double Sample(const Vector3<double>& position) const = 0;
        virtual Vector3<double> Gradient(const Vector3<double>& position) const = 0;
        virtual double Laplacian(const Vector3<double>& position) const = 0;
};

#endif // SCALAR_FIELD_3_HPP