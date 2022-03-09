#ifndef DIFFUSION_SOLVER_HPP
#define DIFFUSION_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"
#include "../grid_systems/fluid_markers.hpp"

class DiffusionSolver
{
    public:
        DiffusionSolver();

        virtual ~DiffusionSolver();

        virtual void Solve(const FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double viscosity, double timeIntervalInSeconds, FaceCenteredGrid3D* output) = 0;;
};

#endif // DIFFUSION_SOLVER_HPP