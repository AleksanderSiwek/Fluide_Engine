#include "single_phase_pressure_solver.hpp"

#include "../linear_system/cuda_jacobi_iteration_solver.hpp"
#include "../linear_system/jacobi_iteration_solver.hpp"
#include "../linear_system/conjugate_gradient_solver.hpp"
#include "../linear_system/cuda_conjugate_gradient_solver.hpp"


SinglePhasePressureSolver::SinglePhasePressureSolver() : _system(LinearSystem())
{
    _systemSolver = std::make_shared<CudaJacobiIterationSolver>(1000, 5, 0.00000000000001);
    //_systemSolver = std::make_shared<JacobiIterationSolver>(1000, 5, 0.00000000000001);
    //_systemSolver = std::make_shared<CudaConjugateGradientSolver>(250, 0.00000000000001);
}

SinglePhasePressureSolver::~SinglePhasePressureSolver()
{

}

void SinglePhasePressureSolver::Solve(FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, double density, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    output->Resize(sourceGrid.GetSize());
    BuildMarkers(fluidSdf, colliderSdf, sourceGrid.GetSize(), sourceGrid);
    BuildSystem(sourceGrid, density, timeIntervalInSeconds);
    _systemSolver->Solve(&_system);
    ApplyPressure(sourceGrid, density, timeIntervalInSeconds, output);
}

void SinglePhasePressureSolver::BuildMarkers(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid)
{
    _fluidMarkers.Resize(size);
    _fluidMarkers.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(fluidSdf.Sample(sourceGrid.GetCellCenterPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = FLUID_MARK;
        }
        else if (colliderSdf.Sample(sourceGrid.GetCellCenterPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = BOUNDRY_MARK;
        }
        else
        {
            _fluidMarkers(i, j, k) = AIR_MARK;
        }
    });
}

Array3<double> SinglePhasePressureSolver::GetPressure() const
{
    return _system.x;
}


void SinglePhasePressureSolver::ApplyPressure(const FaceCenteredGrid3D& input, double density, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    const auto& pressure = _system.x;
    //Vector3<double> scaler = timeIntervalInSeconds / ( density * input.GetGridSpacing());
    Vector3<double> scaler = 1.0 / input.GetGridSpacing();
    Vector3<size_t> size = input.GetSize();

    const auto& inX = input.GetDataXRef();
    const auto& inY = input.GetDataYRef();
    const auto& inZ = input.GetDataZRef();
    auto& outX = output->GetDataXRef();
    auto& outY = output->GetDataYRef();
    auto& outZ = output->GetDataZRef();

    outX.ParallelFill(inX);
    outY.ParallelFill(inY);
    outZ.ParallelFill(inZ);

    _system.x.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(_fluidMarkers(i, j, k) == FLUID_MARK)
        {
            if (i + 1 < size.x && _fluidMarkers(i + 1, j, k) != BOUNDRY_MARK) 
            {
                outX(i + 1, j, k) = inX(i + 1, j, k) + scaler.x * (pressure(i + 1, j, k) - pressure(i, j, k));
            }
            if (j + 1 < size.y && _fluidMarkers(i, j + 1, k) != BOUNDRY_MARK) 
            {
                outY(i, j + 1, k) = inY(i, j + 1, k) + scaler.y * (pressure(i, j + 1, k) - pressure(i, j, k));
            }
            if (k + 1 < size.z && _fluidMarkers(i, j, k + 1) != BOUNDRY_MARK) 
            {
                outZ(i, j, k + 1) = inZ(i, j, k + 1) + scaler.z * (pressure(i, j, k + 1) - pressure(i, j, k));
            }
        }
    });
}

void SinglePhasePressureSolver::SetLinearSystemSolver(const std::shared_ptr<LinearSystemSolver>& solver)
{
    _systemSolver = solver;
}

void SinglePhasePressureSolver::BuildSystem(const FaceCenteredGrid3D& input, double density, double timeIntervalInSeconds)
{
    const auto& size = input.GetSize();
    _system.Resize(size);
    Vector3<double> invH = 1.0 / input.GetGridSpacing();
    Vector3<double> invHSqr = invH * invH;

    auto& A = _system.A;
    auto& b = _system.b;

    _system.A.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        auto& row = A(i, j, k);

        row.center = row.right =  row.up = row.front = 0.0;
        b(i, j, k) = 0.0;

        if(_fluidMarkers(i, j, k) == FLUID_MARK)
        {
            b(i, j, k) = input.DivergenceAtCallCenter(i, j, k);

            if (i + 1 < size.x && _fluidMarkers(i + 1, j, k) != BOUNDRY_MARK) 
            {
                row.center += invHSqr.x;
                if (_fluidMarkers(i + 1, j, k) == FLUID_MARK) 
                {
                    row.right -= invHSqr.x;
                }
            }
            if(i > 0 && _fluidMarkers(i - 1, j, k) != BOUNDRY_MARK)
            {
                row.center += invHSqr.x;
            }
            
            if (j + 1 < size.y && _fluidMarkers(i, j + 1, k) != BOUNDRY_MARK) 
            {
                row.center += invHSqr.y;
                if (_fluidMarkers(i, j + 1, k) == FLUID_MARK) 
                {
                    row.up -= invHSqr.y;
                }
            }
            if(j > 0 && _fluidMarkers(i, j - 1, k) != BOUNDRY_MARK)
            {
                row.center += invHSqr.y;
            }

            if (k + 1 < size.z && _fluidMarkers(i, j, k + 1) != BOUNDRY_MARK) 
            {
                row.center += invHSqr.z;
                if (_fluidMarkers(i, j, k + 1) == FLUID_MARK) 
                {
                    row.front -= invHSqr.z;
                }
            }
            if(k > 0 && _fluidMarkers(i, j, k - 1) != BOUNDRY_MARK)
            {
                row.center += invHSqr.z;
            }
        }
        else
        {
            row.center = 1.0;
        }
    });
}

