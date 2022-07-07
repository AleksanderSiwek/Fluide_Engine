#include <gtest/gtest.h>
#include "../src/linear_system/jacobi_iteration_solver.hpp"
#include "../src/linear_system/cuda_jacobi_iteration_solver.hpp"
#include "../src/grid_systems/face_centered_grid3d.hpp"
#include "../src/grid_systems/fluid_markers.hpp"

// TO DO: Proper tests

void BuildSystem(const FaceCenteredGrid3D& input, const FluidMarkers& markers, LinearSystem* system)
{
    Vector3<size_t> size = input.GetSize();
    system->Resize(size);
    Vector3<double> invH = 1.0 / input.GetGridSpacing();
    Vector3<double> invHSqr = invH * invH;

    auto& A = system->A;
    auto& b = system->b;

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


void PrintSystem(const LinearSystem& system)
{
    Vector3<size_t> size = system.A.GetSize();
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                std::cout << "A(" << i << ", " << j << ", " << k << ") : " << "c = " << system.A(i, j, k).center 
                                                                           << " r = " << system.A(i, j, k).right 
                                                                           << " u = " << system.A(i, j, k).up 
                                                                           << " f = " << system.A(i, j, k).front << "\n";
                std::cout << "b(" << i << ", " << j << ", " << k << ") : " << system.b(i, j, k) << "\n";
                std::cout << "x(" << i << ", " << j << ", " << k << ") : " << system.x(i, j, k) << "\n\n";
            }
        }
    }
}

#include <iostream>
TEST(JacobiIterationSolverTest, Solve_test)
{
    Vector3<size_t> size(20, 20, 20);
    Vector3<double> initialValue(2, 2, 2);
    FaceCenteredGrid3D fluidGrid = FaceCenteredGrid3D(size, 0, 1, initialValue);
    fluidGrid.x(1, 1, 1) = 1;
    fluidGrid.y(1, 1, 1) = 1;
    fluidGrid.z(1, 1, 1) = 1;
    fluidGrid.x(1, 2, 1) = 1;
    fluidGrid.y(1, 2, 1) = 1;
    fluidGrid.z(1, 2, 1) = 1;
    FluidMarkers markers(size, FLUID_MARK);

    LinearSystem system;
    BuildSystem(fluidGrid, markers, &system);
    LinearSystem system_2;
    BuildSystem(fluidGrid, markers, &system_2);

    SystemVector x_expected(2, 2, 2);
    x_expected(0, 0, 0) = 0.5;
    x_expected(1, 0, 0) = 0.5;
    x_expected(0, 0, 1) = 0;
    x_expected(1, 0, 1) = 0;
    x_expected(0, 1, 0) = -0.5;
    x_expected(1, 1, 0) = 0;
    x_expected(0, 1, 1) = 0.5;
    x_expected(1, 1, 1) = 0.5;

    JacobiIterationSolver solver(100, 2, 0.0000001);
    solver.Solve(&system);
    CudaJacobiIterationSolver cudaSolver(100, 2, 0.0000001);
    cudaSolver.Solve(&system_2);

    EXPECT_EQ(true, system.x.IsEqual(system_2.x));
}
