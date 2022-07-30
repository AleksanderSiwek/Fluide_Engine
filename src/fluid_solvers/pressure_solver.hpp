#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"
#include "../grid_systems/fluid_markers.hpp"
#include "../3d/scalar_grid3d.hpp"

class PressureSolver
{
    public:
        PressureSolver();

        virtual ~PressureSolver();

        virtual void Solve(FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, double density, double timeIntervalInSeconds, FaceCenteredGrid3D* output) = 0;
};

#endif // PRESSURE_SOLVER_HPP