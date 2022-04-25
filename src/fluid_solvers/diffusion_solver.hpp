#ifndef DIFFUSION_SOLVER_HPP
#define DIFFUSION_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"
#include "../grid_systems/fluid_markers.hpp"
#include "../3d/scalar_grid3d.hpp"

class DiffusionSolver
{
    public:
        DiffusionSolver();

        virtual ~DiffusionSolver();

        virtual void Solve(const FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, double viscosity, double timeIntervalInSeconds, FaceCenteredGrid3D* output) = 0;;
};

#endif // DIFFUSION_SOLVER_HPP