#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"

class PressureSolver
{
    public:
        PressureSolver(double density=1);

        virtual ~PressureSolver();

        double GetDensity() const;

        void SetDensity(double density);

        virtual void CalculatePressure(FaceCenteredGrid3D& source_grid, const Array3<size_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output) = 0;

    protected: 
        double _density;

};

#endif // PRESSURE_SOLVER_HPP