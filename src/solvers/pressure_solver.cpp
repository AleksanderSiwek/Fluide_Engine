#include "pressure_solver.hpp"

PressureSolver::PressureSolver(double density)
{

}

PressureSolver::~PressureSolver()
{
    
}

double PressureSolver::GetDensity() const
{
    return _density;
}

void PressureSolver::SetDensity(double density)
{
    _density = density;
}