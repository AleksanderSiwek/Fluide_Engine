#ifndef DIFFUSION_SOLVER_HPP
#define DIFFUSION_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"
#include "../grid_systems/fluid_markers.hpp"

class DiffusionSolver
{
    public:
        DiffusionSolver(double viscosity = 0);

        virtual ~DiffusionSolver();

        double GetViscosity() const;

        void SetViscosity(double viscosity);

        virtual void Solve(const FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output) = 0;

    protected:
        double _viscosity;
};

#endif // DIFFUSION_SOLVER_HPP