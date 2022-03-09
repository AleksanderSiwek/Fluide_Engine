#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"
#include "../grid_systems/fluid_markers.hpp"

class PressureSolver
{
    public:
        PressureSolver();

        virtual ~PressureSolver();

        virtual void Solve(FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double density, double timeIntervalInSeconds, FaceCenteredGrid3D* output) = 0;
};

#endif // PRESSURE_SOLVER_HPP