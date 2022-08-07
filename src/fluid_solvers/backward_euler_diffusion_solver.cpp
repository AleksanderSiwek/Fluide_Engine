#include "backward_euler_diffusion_solver.hpp"

#include "../linear_system/cuda_jacobi_iteration_solver.hpp"
#include "../linear_system/cuda_conjugate_gradient_solver.hpp"


BackwardEulerDiffusionSolver::BackwardEulerDiffusionSolver() : _system(LinearSystem())
{
    //_systemSolver = std::make_shared<CudaJacobiIterationSolver>(1000, 5, 0.000001);
    _systemSolver = std::make_shared<CudaConjugateGradientSolver>(250, 0.00000000000001);
}

BackwardEulerDiffusionSolver::~BackwardEulerDiffusionSolver()
{

}

void BackwardEulerDiffusionSolver::Solve(const FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, double viscosity, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    output->Resize(sourceGrid.GetSize());
    Vector3<double> spacing = sourceGrid.GetGridSpacing();
    Vector3<double> c = (timeIntervalInSeconds * viscosity) / (spacing * spacing);

    BuildXMarkers(fluidSdf, colliderSdf, sourceGrid.GetSize(), sourceGrid);
    BuildSystem(sourceGrid.GetDataXRef(), c);
    _systemSolver->Solve(&_system);
    output->GetDataXPtr()->ParallelFill(_system.x);
    
    BuildYMarkers(fluidSdf, colliderSdf, sourceGrid.GetSize(), sourceGrid);
    BuildSystem(sourceGrid.GetDataYRef(), c);
    _systemSolver->Solve(&_system);
    output->GetDataYPtr()->ParallelFill(_system.x);

    BuildZMarkers(fluidSdf, colliderSdf, sourceGrid.GetSize(), sourceGrid);
    BuildSystem(sourceGrid.GetDataZRef(), c);
    _systemSolver->Solve(&_system);
    output->GetDataZPtr()->ParallelFill(_system.x);
}

void BackwardEulerDiffusionSolver::BuildMarkers(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid)
{
    _fluidMarkers.Resize(size);
    _fluidMarkers.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(fluidSdf.Sample(sourceGrid.GridIndexToPosition(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = FLUID_MARK;
        }
        else if(colliderSdf.Sample(sourceGrid.GridIndexToPosition(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = BOUNDRY_MARK;
        }
        else
        {
            _fluidMarkers(i, j, k) = AIR_MARK;
        }
    });
}

void BackwardEulerDiffusionSolver::BuildXMarkers(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid)
{
    _fluidMarkers.Resize(size);
    _fluidMarkers.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(fluidSdf.Sample(sourceGrid.GetXPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = FLUID_MARK;
        }
        else if(colliderSdf.Sample(sourceGrid.GetXPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = BOUNDRY_MARK;
        }
        else
        {
            _fluidMarkers(i, j, k) = AIR_MARK;
        }
    });
}
void BackwardEulerDiffusionSolver::BuildYMarkers(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid)
{
    _fluidMarkers.Resize(size);
    _fluidMarkers.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(fluidSdf.Sample(sourceGrid.GetYPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = FLUID_MARK;
        }
        else if(colliderSdf.Sample(sourceGrid.GetYPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = BOUNDRY_MARK;
        }
        else
        {
            _fluidMarkers(i, j, k) = AIR_MARK;
        }
    });
}

void BackwardEulerDiffusionSolver::BuildZMarkers(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid)
{
    _fluidMarkers.Resize(size);
    _fluidMarkers.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(fluidSdf.Sample(sourceGrid.GetZPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = FLUID_MARK;
        }
        else if(colliderSdf.Sample(sourceGrid.GetZPos(i, j, k)) < 0)
        {
            _fluidMarkers(i, j, k) = BOUNDRY_MARK;
        }
        else
        {
            _fluidMarkers(i, j, k) = AIR_MARK;
        }
    });
}

void BackwardEulerDiffusionSolver::BuildSystem(const Array3<double>& arr, Vector3<double> c)
{
    BuildMatrix(arr.GetSize(), c);
    BuildVectors(arr, c);
}

void BackwardEulerDiffusionSolver::BuildMatrix(Vector3<size_t> size, Vector3<double> c)
{
    _system.A.Resize(size);

    _system.A.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        auto& row = _system.A(i, j, k);

        row.center = 1.0;
        row.up = row.front = row.right = 0.0;

        if(_fluidMarkers(i, j, k) == FLUID_MARK)
        {
            if(i + 1 < size.x)
            {
                if(_fluidMarkers(i + 1, j, k) != AIR_MARK)
                {
                    row.center += c.x;
                }
                if(_fluidMarkers(i + 1, j, k) == FLUID_MARK)
                {
                    row.right -= c.x;
                }
            }
            if(i > 0 && _fluidMarkers(i - 1, j, k) != AIR_MARK)
            {
                row.center += c.x;
            }

            if(j + 1 < size.y)
            {
                if(_fluidMarkers(i, j + 1, k) != AIR_MARK)
                {
                    row.center += c.y;
                }
                if(_fluidMarkers(i, j + 1, k) == FLUID_MARK)
                {
                    row.up -= c.y;
                }
            }
            if(j > 0 && _fluidMarkers(i, j - 1, k) != AIR_MARK)
            {
                row.center += c.y;
            }

            if(k + 1 < size.z)
            {
                if(_fluidMarkers(i, j, k + 1) != AIR_MARK)
                {
                    row.center += c.z;
                }
                if(_fluidMarkers(i, j, k + 1) == FLUID_MARK)
                {
                    row.front -= c.z;
                }
            }
            if(k > 0 && _fluidMarkers(i, j, k - 1) != AIR_MARK)
            {
                row.center += c.z;
            }
        }
    });
}

void BackwardEulerDiffusionSolver::BuildVectors(const Array3<double>& arr, Vector3<double> c)
{
    const auto& size = arr.GetSize();
    _system.x.Resize(size);
    _system.b.Resize(size);

    _system.x.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        auto& b = _system.b(i, j, k);
        auto& x = _system.x(i, j, k);
        b = x = arr(i, j, k);

        if(_fluidMarkers(i, j, k) == FLUID_MARK)
        {
            if(i + 1 < size.x && _fluidMarkers(i + 1, j, k) == BOUNDRY_MARK)
            {
                b += c.x * arr(i + 1, j, k);
            }
            if(i > 0 && _fluidMarkers(i - 1, j, k) == BOUNDRY_MARK)
            {
                b += c.x * arr(i - 1, j, k);
            }

            if(j + 1 < size.y && _fluidMarkers(i, j + 1, k) == BOUNDRY_MARK)
            {
                b += c.y * arr(i, j + 1, k);
            }
            if(j > 0 && _fluidMarkers(i, j - 1, k) == BOUNDRY_MARK)
            {
                b += c.y * arr(i, j - 1, k);
            }

            if(k + 1 < size.z && _fluidMarkers(i, j, k + 1) == BOUNDRY_MARK)
            {
                b += c.z * arr(i, j, k + 1);
            }
            if(k > 0 && _fluidMarkers(i, j, k - 1) == BOUNDRY_MARK)
            {                
                b += c.z * arr(i, j, k - 1);     
            }
        }
    });
}


