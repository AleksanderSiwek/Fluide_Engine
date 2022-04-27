#ifndef _BACKWARD_EULER_DIFFUSION_SOLVER_HPP
#define _BACKWARD_EULER_DIFFUSION_SOLVER_HPP

#include <memory>
#include "diffusion_solver.hpp"
#include "../linear_system/linear_system.hpp"
#include "../linear_system/jacobi_iteration_solver.hpp"


class BackwardEulerDiffusionSolver : public DiffusionSolver
{
    public:
        BackwardEulerDiffusionSolver();

        ~BackwardEulerDiffusionSolver();

        void Solve(const FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, double viscosity, double timeIntervalInSeconds, FaceCenteredGrid3D* output) override;

    private:
        FluidMarkers _fluidMarkers;
        LinearSystem _system;
        std::shared_ptr<LinearSystemSolver> _systemSolver;

        void BuildMarkers(const ScalarGrid3D& fluidSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid);
        void BuildSystem(const Array3<double>& arr, Vector3<double> c);
        void BuildMatrix(Vector3<size_t> size, Vector3<double> c);
        void BuildVectors(const Array3<double>& arr, Vector3<double> c);
};

#endif // _BACKWARD_EULER_DIFFUSION_SOLVER_HPP