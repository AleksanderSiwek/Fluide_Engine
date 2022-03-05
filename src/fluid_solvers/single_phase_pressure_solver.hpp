#ifndef _SINGLE_PHASE_PRESSURE_SOLVER
#define _SINGLE_PHASE_PRESSURE_SOLVER

#include "pressure_solver.hpp"


class SinglePhasePressureSolver : public PressureSolver
{
    public:
        SinglePhasePressureSolver();
        
        ~SinglePhasePressureSolver();

        virtual FaceCenteredGrid3D CalculatePressure(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds) override;
        virtual void CalculatePressure(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* out) = 0;

    private:

};

#endif _SINGLE_PHASE_PRESSURE_SOLVER