#ifndef _SINGLE_PHASE_PRESSURE_SOLVER
#define _SINGLE_PHASE_PRESSURE_SOLVER

#include "pressure_solver.hpp"
#include "../linear_system/jacobi_iteration_solver.hpp"

// TO DO: Change _systemSolver to pointer to astract LinearSystemSolver
// TO DO: REFACTOR!!!!!

class SinglePhasePressureSolver : public PressureSolver
{
    public:
        SinglePhasePressureSolver();
        
        ~SinglePhasePressureSolver();

        virtual void CalculatePressure(FaceCenteredGrid3D& source_grid, const Array3<size_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output) override;
        void ApplyPressure(const FaceCenteredGrid3D& input, const Array3<size_t>& fluidMarkers, FaceCenteredGrid3D* output);

    private:
        LinearSystem _system;
        JacobiIterationSolver _systemSolver;

};

#endif // _SINGLE_PHASE_PRESSURE_SOLVER