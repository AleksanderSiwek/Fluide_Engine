#include "single_phase_pressure_solver.hpp"


SinglePhasePressureSolver::SinglePhasePressureSolver() : _system(LinearSystem()), _systemSolver(JacobiIterationSolver(1000, 5, 0.000001))
{

}

SinglePhasePressureSolver::~SinglePhasePressureSolver()
{

}

void SinglePhasePressureSolver::CalculatePressure(FaceCenteredGrid3D& source_grid, const Array3<size_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    _system.Build(source_grid, fluidMarkers);
    _systemSolver.Solve(&_system);
    ApplyPressure(source_grid, fluidMarkers, output);
}

void SinglePhasePressureSolver::ApplyPressure(const FaceCenteredGrid3D& input, const Array3<size_t>& fluidMarkers, FaceCenteredGrid3D* output)
{
    const auto& pressure = _system.x;
    Vector3<double> invH = 1.0 / input.GetGridSpacing();
    Vector3<size_t> size = input.GetSize();

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                // TO DO 
            }
        }   
    }
}
