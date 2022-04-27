#include "backward_euler_diffusion_solver.hpp"

BackwardEulerDiffusionSolver::BackwardEulerDiffusionSolver() : _system(LinearSystem())
{
    _systemSolver = std::make_shared<JacobiIterationSolver>(1000, 1, 0.0000000001);
}

BackwardEulerDiffusionSolver::~BackwardEulerDiffusionSolver()
{

}

#include <iostream>
void BackwardEulerDiffusionSolver::Solve(const FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, double viscosity, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    output->Resize(sourceGrid.GetSize());
    BuildMarkers(fluidSdf, sourceGrid.GetSize(), sourceGrid);
    Vector3<double> spacing = sourceGrid.GetGridSpacing();
    Vector3<double> c = (timeIntervalInSeconds * viscosity) / (spacing * spacing);

    std::cout << "dt = " << timeIntervalInSeconds << "\n";
    std::cout << "viscosity = " << viscosity << "\n";
    std::cout << "grid spacing = " << (spacing * spacing).x << ", " << (spacing * spacing).y << ", " << (spacing * spacing).z << "\n";
    std::cout << "c = " << c.x << ", " << c.y << ", " << c.z << "\n";
    const auto& size = sourceGrid.GetSize();
    std::cout << "Size Diffusion Solver: (" << size.x << ", " << size.y << ", " << size.z << ")\n";
    for(size_t j = size.y ; j > 0; j--)
    {
        for(size_t k = 0; k < size.z; k++)
        {
            for(size_t i = 0; i < size.x; i++)
            {
                std::cout << (_fluidMarkers(i, j - 1, k) == FLUID_MARK ? "F" : "A") << " ";
            }
            std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    BuildSystem(sourceGrid.GetDataXRef(), c);
    _systemSolver->Solve(&_system);
    output->GetDataXPtr()->Fill(_system.x);

    // BuildSystem(sourceGrid.GetDataYRef(), c);
    // _systemSolver->Solve(&_system);
    // output->GetDataYPtr()->Fill(_system.x);

    // BuildSystem(sourceGrid.GetDataZRef(), c);
    // _systemSolver->Solve(&_system);
    // output->GetDataZPtr()->Fill(_system.x);
}

void BackwardEulerDiffusionSolver::BuildMarkers(const ScalarGrid3D& fluidSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid)
{
    _fluidMarkers.Resize(size);
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                if(fluidSdf.Sample(sourceGrid.GridIndexToPosition(i, j, k)) < 0)
                {
                    _fluidMarkers(i, j, k) = FLUID_MARK;
                }
                else
                {
                    _fluidMarkers(i, j, k) = AIR_MARK;
                }
            }
        }
    }
}

void BackwardEulerDiffusionSolver::BuildSystem(const Array3<double>& arr, Vector3<double> c)
{
    BuildMatrix(arr.GetSize(), c);
    BuildVectors(arr, c);

    // std::cout << "System build\n";
    // const auto& size = arr.GetSize();
    // size_t sizeVectorized = size.x * size.y * size.z;
    // const auto& A = _system.A.GetRawData();
    // const auto& x = _system.x.GetRawData();
    // const auto& b = _system.b.GetRawData();
    // std::cout << "A.ce A.ri A.up A.do *   x = b\n";
    // for(size_t i = 0; i < sizeVectorized; i++)
    // {
    //     std::cout << A[i].center << ", " << A[i].right << ", " << A[i].up << ", " << A[i].front << "  *  ";
    //     std::cout << x[i] << " = " << b[i] << "\n";
    // }
    // std::cout << "\n";
}

// TO DO: Check dirichlet and neuman boundry types
// This is dirichhlet type of boundry

void BackwardEulerDiffusionSolver::BuildMatrix(Vector3<size_t> size, Vector3<double> c)
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
            }
        }
    }
}

void BackwardEulerDiffusionSolver::BuildVectors(const Array3<double>& arr, Vector3<double> c)
{
    const auto& size = arr.GetSize();
    std::cout << "Size BuildVectors: (" << size.x << ", " << size.y << ", " << size.z << ")\n";

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
            }
        }
    }
}


