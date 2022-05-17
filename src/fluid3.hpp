#ifndef _FLUID_HPP
#define _FLUID_HPP

#include "grid_systems/face_centered_grid3d.hpp"
#include "particle_systems/particle_system.hpp"
#include "grid_systems/fluid_markers.hpp"
#include "3d/scalar_grid3d.hpp"


class Fluid3
{
    public:
        Fluid3();

        ~Fluid3();

        FaceCenteredGrid3D velocityGrid;
        FluidMarkers markers;
        Array3<int> xMarkers;
        Array3<int> yMarkers;
        Array3<int> zMarkers;
        ScalarGrid3D sdf;
        ParticleSystem particleSystem;
        double viscosity;
        double density;
};


#endif // _FLUID_HPP