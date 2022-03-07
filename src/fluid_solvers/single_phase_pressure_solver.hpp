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

        virtual void CalculatePressure(FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output) override;
        void ApplyPressure(const FaceCenteredGrid3D& input, const FluidMarkers& fluidMarkers, FaceCenteredGrid3D* output);

        void SetLinearSystemSolver(const std::shared_ptr<LinearSystemSolver>& solver);

    private:
        LinearSystem _system;
        std::shared_ptr<LinearSystemSolver> _systemSolver;

        void BuildSystem(const FaceCenteredGrid3D& input, const FluidMarkers& markers);

};

#endif // _SINGLE_PHASE_PRESSURE_SOLVER