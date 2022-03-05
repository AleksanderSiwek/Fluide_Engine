#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"

class PressureSolver
{
    public:
        PressureSolver(double density);

        virtual ~PressureSolver();

        double GetDensity() const;

        void SetDensity(double density);

        virtual FaceCenteredGrid3D CalculatePressure(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds) = 0;
        virtual void CalculatePressure(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* out) = 0;

    protected: 
        double _density;

};

#endif // PRESSURE_SOLVER_HPP