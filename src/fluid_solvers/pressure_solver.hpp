#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"
#include "../grid_systems/fluid_markers.hpp"

class PressureSolver
{
    public:
        PressureSolver(double density=1);

        virtual ~PressureSolver();

        double GetDensity() const;

        void SetDensity(double density);

        virtual void CalculatePressure(FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output) = 0;

    protected: 
        double _density;

};

#endif // PRESSURE_SOLVER_HPP