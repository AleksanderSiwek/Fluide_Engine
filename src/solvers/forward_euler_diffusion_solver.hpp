#ifndef _FORWARD_EULER_DIFFUSION_SOLVER_HPP
#define _FORWARD_EULER_DIFFUSION_SOLVER_HPP

#include "diffusion_solver.hpp"


class ForwardEulerDiffusionSolver : public DiffusionSolver
{
    public:
        ForwardEulerDiffusionSolver();

        ~ForwardEulerDiffusionSolver();  

        Array3<double> CalculateDiffusion(const Array3<Vector3<double>>& source, double diffusionCoefficient, double timeIntervalInSeconds, Vector3<double> gridSpacing, Vector3<double> dataPosition) override;
        void CalculateDiffusion(const Array3<Vector3<double>>& source, double diffusionCoefficient, double timeIntervalInSeconds, Vector3<double> gridSpacing, Vector3<double> dataPosition, Array3<double>* dest) override;

    private:
};

#endif // _FORWARD_EULER_DIFFUSION_SOLVER_HPP