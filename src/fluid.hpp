#ifndef _FLUID_HPP
#define _FLUID_HPP

#include "grid_systems/face_centered_grid3d.hpp"
#include "particle_systems/particle_system.hpp"

// TO DO: ? Well see

class Fluid
{
    public:
        Fluid();

        ~Fluid();

        FaceCenteredGrid3D velocityGrid;
        ParticleSystem particleSystem;

};


#endif // _FLUID_HPP