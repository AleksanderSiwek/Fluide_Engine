#include <gtest/gtest.h>
#include "../src/linear_system/jacobi_iteration_solver.hpp"
#include "../src/grid_systems/face_centered_grid3d.hpp"

// TO DO: Proper tests

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
                std::cout << "b(" << i << ", " << j << ", " << k << ") : " << system.b(i, j, k) << "\n\n";
            }
        }
    }
}

TEST(JacobiIterationSolverTest, Solve_test)
{
    Vector3<size_t> size(4, 4, 4);
    Vector3<double> initialValue(2, 2, 2);
    FaceCenteredGrid3D fluidGrid = FaceCenteredGrid3D(size, 0, 1, initialValue);
    fluidGrid.x(1, 1, 1) = 1;
    fluidGrid.y(1, 1, 1) = 1;
    fluidGrid.z(1, 1, 1) = 1;
    fluidGrid.x(1, 2, 1) = 1;
    fluidGrid.y(1, 2, 1) = 1;
    fluidGrid.z(1, 2, 1) = 1;
    Array3<size_t> markers(size, 1);
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                if(i == 0 || j == 0 || k == 0 || i == size.x - 1 || j == size.y - 1 || k == size.z - 1)
                    markers(i, j, k) = 2;
            }
        }
    }

    LinearSystem system;
    system.Build(fluidGrid, markers);

    SystemVector x_expected(2, 2, 2);
    x_expected(0, 0, 0) = 0.5;
    x_expected(1, 0, 0) = 0.5;
    x_expected(0, 0, 1) = 0;
    x_expected(1, 0, 1) = 0;
    x_expected(0, 1, 0) = -0.5;
    x_expected(1, 1, 0) = 0;
    x_expected(0, 1, 1) = 0.5;
    x_expected(1, 1, 1) = 0.5;

    JacobiIterationSolver solver(100, 1, 0.001);
    solver.Solve(&system);
    EXPECT_EQ(true, system.x.IsEqual(x_expected));
}
