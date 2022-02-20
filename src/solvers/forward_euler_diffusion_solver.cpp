#include "forward_euler_diffusion_solver.hpp"

ForwardEulerDiffusionSolver::ForwardEulerDiffusionSolver()
{

}

ForwardEulerDiffusionSolver::~ForwardEulerDiffusionSolver()
{

}

Array3<double> ForwardEulerDiffusionSolver::CalculateDiffusion(const Array3<Vector3<double>>& source, double diffusionCoefficient, double timeIntervalInSeconds, Vector3<double> gridSpacing, Vector3<double> dataPosition)
{
    
    return 1;
}

void ForwardEulerDiffusionSolver::CalculateDiffusion(const Array3<Vector3<double>>& source, double diffusionCoefficient, double timeIntervalInSeconds, Vector3<double> gridSpacing, Vector3<double> dataPosition, Array3<double>* dest)
{
    Array3<double>* out = new Array3<double>(1);
    dest = out;
}
