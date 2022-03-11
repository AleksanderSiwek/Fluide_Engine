#ifndef EXTERNAL_FORCE_HPP
#define EXTERNAL_FORCE_HPP

#include "../common/array3.hpp"


class ExternalForce
{
    public:
        ExternalForce() {}

        virtual ~ExternalForce() {}

        Vector3<double> ApplyExternalForce(Array3<double>& values)
        {
            return OnApplyExternalForce(values);
        }

    protected:
        virtual Vector3<double> OnApplyExternalForce(Array3<double>& values) = 0;
};

#endif // EXTERNAL_FORCE_HPP