#ifndef DIFFUSION_SOLVER_HPP
#define DIFFUSION_SOLVER_HPP

#include "../common/array3.hpp"

class DiffusionSolver
{
    public:
        DiffusionSolver(double viscosity);

        virtual ~DiffusionSolver();

        double GetViscosity() const;

        void SetViscosity(double viscosity);

        virtual Array3<double> CalculateDiffusion(Array3<Vector3<double>> velocity) = 0;

    protected:
        double _viscosity;

};

#endif // DIFFUSION_SOLVER_HPP