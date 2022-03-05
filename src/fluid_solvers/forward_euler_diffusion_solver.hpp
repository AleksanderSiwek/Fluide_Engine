#ifndef _FORWARD_EULER_DIFFUSION_SOLVER_HPP
#define _FORWARD_EULER_DIFFUSION_SOLVER_HPP

#include "diffusion_solver.hpp"


class ForwardEulerDiffusionSolver : public DiffusionSolver
{
    public:
        ForwardEulerDiffusionSolver();

        ~ForwardEulerDiffusionSolver();  

        FaceCenteredGrid3D CalculateDiffusion(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds) override;
        void CalculateDiffusion(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* dest_grid) override;
        double CalculateLaplacian(Array3<double>* grid, const Array3<uint8_t> fluidMarkers, Vector3<double> gridSpacing, size_t i, size_t j, size_t k);

    private:
};

#endif // _FORWARD_EULER_DIFFUSION_SOLVER_HPP