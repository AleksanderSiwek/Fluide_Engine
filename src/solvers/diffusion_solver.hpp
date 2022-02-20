#ifndef DIFFUSION_SOLVER_HPP
#define DIFFUSION_SOLVER_HPP

#include "../common/array3.hpp"

class DiffusionSolver
{
    public:
        DiffusionSolver(double viscosity = 0);

        virtual ~DiffusionSolver();

        double GetViscosity() const;

        void SetViscosity(double viscosity);

        virtual Array3<double> CalculateDiffusion(const Array3<Vector3<double>>& source, double diffusionCoefficient, double timeIntervalInSeconds, Vector3<double> gridSpacing, Vector3<double> dataPosition) = 0;
        virtual void CalculateDiffusion(const Array3<Vector3<double>>& source, double diffusionCoefficient, double timeIntervalInSeconds, Vector3<double> gridSpacing, Vector3<double> dataPosition, Array3<double>* dest) = 0;

    protected:
        double _viscosity;

};

#endif // DIFFUSION_SOLVER_HPP