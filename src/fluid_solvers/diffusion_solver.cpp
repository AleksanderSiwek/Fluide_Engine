#include "diffusion_solver.hpp"

DiffusionSolver::DiffusionSolver(double viscosity) : _viscosity(viscosity)
{

}

DiffusionSolver::~DiffusionSolver()
{

}

double DiffusionSolver::GetViscosity() const
{
    return _viscosity;
}

void DiffusionSolver::SetViscosity(double viscosity)
{
    _viscosity = viscosity;
}