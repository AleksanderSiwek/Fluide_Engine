#ifndef EXTERNAL_FORCE_HPP
#define EXTERNAL_FORCE_HPP

#include "../grid_systems/face_centered_grid3d.hpp"


class ExternalForce
{
    public:
        ExternalForce() {}

        virtual ~ExternalForce() {}

        virtual void ApplyExternalForce(FaceCenteredGrid3D& velGrid, const double timeIntervalInSeconds);
        virtual Vector3<double> Sample(const Vector3<double>& position) const = 0;
};

#endif // EXTERNAL_FORCE_HPP