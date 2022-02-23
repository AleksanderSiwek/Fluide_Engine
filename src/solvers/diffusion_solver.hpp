#ifndef DIFFUSION_SOLVER_HPP
#define DIFFUSION_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"

class DiffusionSolver
{
    public:
        DiffusionSolver(double viscosity = 0);

        virtual ~DiffusionSolver();

        double GetViscosity() const;

        void SetViscosity(double viscosity);

        virtual FaceCenteredGrid3D CalculateDiffusion(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds) = 0;
        virtual void CalculateDiffusion(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* dest_grid) = 0;

    protected:
        double _viscosity;
};

#endif // DIFFUSION_SOLVER_HPP