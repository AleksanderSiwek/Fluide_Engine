#ifndef _FLUID_HPP
#define _FLUID_HPP

#include "grid_systems/face_centered_grid3d.hpp"
#include "particle_systems/particle_system.hpp"


class Fluid3
{
    public:
        Fluid3();

        ~Fluid3();

        FaceCenteredGrid3D velocityGrid;
        ParticleSystem particleSystem;
        double viscosity;
        double density;
};


#endif // _FLUID_HPP