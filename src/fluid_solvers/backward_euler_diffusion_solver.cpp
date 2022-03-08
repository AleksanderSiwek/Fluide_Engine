#include "backward_euler_diffusion_solver.hpp"

BackwardEulerDiffusionSolver::BackwardEulerDiffusionSolver() : _system(LinearSystem())
{
    _systemSolver = std::make_shared<JacobiIterationSolver>(1000, 5, 0.000001);
}

BackwardEulerDiffusionSolver::~BackwardEulerDiffusionSolver()
{

}

void BackwardEulerDiffusionSolver::Solve(const FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    output->Resize(source_grid.GetSize());
    Vector3<double> spacing = source_grid.GetGridSpacing();
    Vector3<double> c = timeIntervalInSeconds * _viscosity / (spacing * spacing);

    BuildSystem(source_grid.GetDataXRef(), c, fluidMarkers);
    _systemSolver->Solve(&_system);
    output->GetDataXPtr()->Fill(_system.x);

    BuildSystem(source_grid.GetDataYRef(), c, fluidMarkers);
    _systemSolver->Solve(&_system);
    output->GetDataYPtr()->Fill(_system.x);

    BuildSystem(source_grid.GetDataZRef(), c, fluidMarkers);
    _systemSolver->Solve(&_system);
    output->GetDataZPtr()->Fill(_system.x);
}

void BackwardEulerDiffusionSolver::BuildSystem(const Array3<double>& arr, Vector3<double> c, const FluidMarkers& fluidMarkers)
{
    BuildMatrix(arr.GetSize(), c, fluidMarkers);
    BuildVectors(arr, c, fluidMarkers);
}

// TO DO: Check dirichlet and neuman boundry types
// This is dirichhlet type of boundry

void BackwardEulerDiffusionSolver::BuildMatrix(Vector3<size_t> size, Vector3<double> c, const FluidMarkers& fluidMarkers)
{
    _system.A.Resize(size);

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                auto& row = _system.A(i, j, k);

                row.center = 1.0;
                row.up = row.front = row.right = 0.0;

                if(fluidMarkers(i, j, k) == FLUID_MARK)
                {
                    if(i + 1 < size.x)
                    {
                        if(fluidMarkers(i +  1, j, k) != AIR_MARK)
                            row.center += c.x;
                        if(fluidMarkers(i +  1, j, k) == FLUID_MARK)
                            row.right -= c.x;
                    }
                    if(i > 0 && fluidMarkers(i - 1, j, k) != AIR_MARK)
                        row.center += c.x;

                    if(j + 1 < size.y)
                    {
                        if(fluidMarkers(i, j + 1, k) != AIR_MARK)
                            row.center += c.y;
                        if(fluidMarkers(i, j + 1, k) == FLUID_MARK)
                            row.up -= c.y;
                    }
                    if(j > 0 && fluidMarkers(i, j - 1, k) != AIR_MARK)
                        row.center += c.y;

                    if(k + 1 < size.z)
                    {
                        if(fluidMarkers(i, j, k + 1) != AIR_MARK)
                            row.center += c.z;
                        if(fluidMarkers(i, j, k + 1) == FLUID_MARK)
                            row.front -= c.z;
                    }
                    if(k > 0 && fluidMarkers(i, j, k - 1) != AIR_MARK)
                        row.center += c.z;
                }
            }
        }
    }
}

void BackwardEulerDiffusionSolver::BuildVectors(const Array3<double>& arr, Vector3<double> c, const FluidMarkers& fluidMarkers)
{
    Vector3<size_t> size = arr.GetSize();

    _system.x.Resize(size);
    _system.b.Resize(size);

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                auto& b = _system.b(i, j, k);
                auto& x = _system.x(i, j, k);
                b = x = arr(i, j, k);

                if(fluidMarkers(i, j, k) == FLUID_MARK)
                {
                    if(i + 1 < size.x && fluidMarkers(i + 1, j, k) == BOUNDRY_MARK)
                        b += c.x * arr(i + 1, j, k);
                    if(i > 0 && fluidMarkers(i - 1, j, k) == BOUNDRY_MARK)
                        b += c.x * arr(i - 1, j, k);

                    if(j + 1 < size.y && fluidMarkers(i, j + 1, k) == BOUNDRY_MARK)
                        b += c.y * arr(i, j + 1, k);
                    if(j > 0 && fluidMarkers(i, j - 1, k) == BOUNDRY_MARK)
                        b += c.y * arr(i, j - 1, k);

                    if(k + 1 < size.x && fluidMarkers(i, j, k + 1) == BOUNDRY_MARK)
                        b += c.z * arr(i, j, k + 1);
                    if(k > 0 && fluidMarkers(i, j, k - 1) == BOUNDRY_MARK)
                        b += c.z * arr(i, j, k - 1);     
                }
            }
        }
    }
}
