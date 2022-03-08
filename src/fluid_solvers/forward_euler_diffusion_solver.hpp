#ifndef _FORWARD_EULER_DIFFUSION_SOLVER_HPP
#define _FORWARD_EULER_DIFFUSION_SOLVER_HPP

#include "diffusion_solver.hpp"


class ForwardEulerDiffusionSolver : public DiffusionSolver
{
    public:
        ForwardEulerDiffusionSolver();

        ~ForwardEulerDiffusionSolver();  

        void Solve(const FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output) override;
        double CalculateLaplacian(const Array3<double>& grid, const FluidMarkers& fluidMarkers, Vector3<double> gridSpacing, size_t i, size_t j, size_t k);

    private:
};

#endif // _FORWARD_EULER_DIFFUSION_SOLVER_HPP