#ifndef _SINGLE_PHASE_PRESSURE_SOLVER
#define _SINGLE_PHASE_PRESSURE_SOLVER

#include <memory>
#include "pressure_solver.hpp"
#include "../linear_system/jacobi_iteration_solver.hpp"

// TO DO: Change _systemSolver to pointer to astract LinearSystemSolver
// TO DO: REFACTOR!!!!!

class SinglePhasePressureSolver : public PressureSolver
{
    public:
        SinglePhasePressureSolver();
        
        ~SinglePhasePressureSolver();

        virtual void Solve(FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, double density, double timeIntervalInSeconds, FaceCenteredGrid3D* output) override;
        void BuildMarkers(const ScalarGrid3D& fluidSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid);
        void ApplyPressure(const FaceCenteredGrid3D& input, double density, double timeIntervalInSeconds, FaceCenteredGrid3D* output);
        
        void SetLinearSystemSolver(const std::shared_ptr<LinearSystemSolver>& solver);
        
    private:
        LinearSystem _system;
        std::shared_ptr<LinearSystemSolver> _systemSolver;
        FluidMarkers _fluidMarkers;

        void BuildSystem(const FaceCenteredGrid3D& input, double density, double timeIntervalInSeconds);
};

#endif // _SINGLE_PHASE_PRESSURE_SOLVER