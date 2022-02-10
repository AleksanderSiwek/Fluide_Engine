#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "../common/array3.hpp"

class PressureSolver
{
    public:
        PressureSolver(double density);

        virtual ~PressureSolver();

        double GetDensity() const;

        void SetDensity(double density);

        virtual Vector3<double> CalculatePressureGradient(Array3<Vector3<double>> pressure) = 0;

    protected: 
        double _density;

};

#endif // PRESSURE_SOLVER_HPP