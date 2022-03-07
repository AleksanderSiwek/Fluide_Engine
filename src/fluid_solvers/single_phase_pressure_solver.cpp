#include "single_phase_pressure_solver.hpp"


SinglePhasePressureSolver::SinglePhasePressureSolver() : _system(LinearSystem())
{
    _systemSolver = std::make_shared<JacobiIterationSolver>(1000, 5, 0.000001);
}

SinglePhasePressureSolver::~SinglePhasePressureSolver()
{

}

void SinglePhasePressureSolver::CalculatePressure(FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    output->Resize(source_grid.GetSize());
    BuildSystem(source_grid, fluidMarkers);
    _systemSolver->Solve(&_system);
    ApplyPressure(source_grid, fluidMarkers, output);
}

void SinglePhasePressureSolver::ApplyPressure(const FaceCenteredGrid3D& input, const FluidMarkers& fluidMarkers, FaceCenteredGrid3D* output)
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

void SinglePhasePressureSolver::SetLinearSystemSolver(const std::shared_ptr<LinearSystemSolver>& solver)
{
    _systemSolver = solver;
}

void SinglePhasePressureSolver::BuildSystem(const FaceCenteredGrid3D& input, const FluidMarkers& markers)
{
    Vector3<size_t> size = input.GetSize();
    _system.Resize(size);
    Vector3<double> invH = 1.0 / input.GetGridSpacing();
    Vector3<double> invHSqr = invH * invH;

    auto& A = _system.A;
    auto& b = _system.b;

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                auto& row = A(i, j, k);

                row.center = row.right =  row.up = row.front = 0.0;
                b(i, j, k) = 0.0;

                if(markers(i, j, k) == FLUID_MARK)
                {
                    b(i, j, k) = input.DivergenceAtCallCenter(i, j, k);

                    if(i + 1 < size.x && markers(i + 1, j, k) != BOUNDRY_MARK)
                    {
                        row.center += invHSqr.x;
                        if(markers(i + 1, j, k) == FLUID_MARK)
                            row.right -= invHSqr.x;
                    }
                    if(i > 0 && markers(i - 1, j, k) != BOUNDRY_MARK)
                        row.center += invHSqr.x;

                    if(j + 1 < size.y && markers(i , j + 1, k) != BOUNDRY_MARK)
                    {
                        row.center += invHSqr.y;
                        if(markers(i, j + 1, k) == FLUID_MARK)
                            row.up -= invHSqr.y;
                    }
                    if(j > 0 && markers(i, j - 1, k) != BOUNDRY_MARK)
                        row.center += invHSqr.y;

                    if(k + 1 < size.z && markers(i , j, k + 1) != BOUNDRY_MARK)
                    {
                        row.center += invHSqr.z;
                        if(markers(i, j, k + 1) == FLUID_MARK)
                            row.front -= invHSqr.z;
                    }
                    if(k > 0 && markers(i, j, k - 1) != BOUNDRY_MARK)
                        row.center += invHSqr.z;
                }
                else
                {
                    row.center = 1.0;
                }
            }
        }
    }
}

