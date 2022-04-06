#ifndef EXTERNAL_FORCE_HPP
#define EXTERNAL_FORCE_HPP

#include "../grid_systems/face_centered_grid3d.hpp"


class ExternalForce
{
    public:
        ExternalForce() {}

        virtual ~ExternalForce() {}

        void ApplyExternalForce(FaceCenteredGrid3D& values)
        {
            OnApplyExternalForce(values);
        }

    protected:
        virtual void OnApplyExternalForce(FaceCenteredGrid3D& values) = 0;
};

#endif // EXTERNAL_FORCE_HPP