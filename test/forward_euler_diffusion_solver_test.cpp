#include <gtest/gtest.h>
#include "../src/fluid_solvers/forward_euler_diffusion_solver.hpp"
#include <iomanip>


void PrintArray3(const Array3<double>& input)
{
    const auto& size = input.GetSize();
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Size: (" << size.x << ", " << size.y << ", " << size.z << ")\n";
    for(size_t j = size.y ; j > 0; j--)
    {
        for(size_t k = 0; k < size.z; k++)
        {
            for(size_t i = 0; i < size.x; i++)
            {
                std::cout << input(i, j - 1, k) << " ";
            }
            std::cout << "      ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

TEST(ForwardEulerDiffusionSolverTest, CalculateLaplacian_test)
{
    const Vector3<size_t> size(3, 3, 3);
    FaceCenteredGrid3D input(size, 0, 1, 1);
    ScalarGrid3D sdf(size, 1, Vector3<double>(0, 0, 0));
    ForwardEulerDiffusionSolver solver;

    auto& xData = input.GetDataXRef();
    auto& yData = input.GetDataYRef();
    auto& zData = input.GetDataZRef();

    xData(1, 1, 1) = yData(1, 1, 1) = zData(1, 1, 1) = 4;
    sdf(1, 1, 1) = -10;

    PrintArray3(input.GetDataXRef());

    FaceCenteredGrid3D output;
    solver.BuildMarkers(sdf, size, input);
    double laplacian = solver.CalculateLaplacian(xData, input.GetGridSpacing(), 1, 1, 1);

    PrintArray3(output.GetDataXRef());
}

TEST(ForwardEulerDiffusionSolverTest, Solve_test)
{
    const Vector3<size_t> size(6, 6, 3);
    FaceCenteredGrid3D input(size, 0, 1, 1);
    ScalarGrid3D sdf(size, 1, Vector3<double>(0, 0, 0));
    ScalarGrid3D sdfCollider(size, 1, Vector3<double>(0, 0, 0));
    ForwardEulerDiffusionSolver solver;

    auto& xData = input.GetDataXRef();
    auto& yData = input.GetDataYRef();
    auto& zData = input.GetDataZRef();

    xData(2, 2, 1) = yData(2, 2, 1) = zData(2, 2, 1) = 3;
    xData(3, 2, 1) = yData(3, 2, 1) = zData(3, 2, 1) = 3;
    xData(2, 3, 1) = yData(2, 3, 1) = zData(2, 3, 1) = 5;
    xData(3, 3, 1) = yData(3, 3, 1) = zData(3, 3, 1) = 5;

    sdf(2, 2, 1) = -1;
    sdf(3, 2, 1) = -1;
    sdf(2, 3, 1) = -1;
    sdf(3, 3, 1) = -1;
    sdf(2, 2, 2) = -1;
    sdf(3, 2, 2) = -1;
    sdf(2, 3, 2) = -1;
    sdf(3, 3, 2) = -1;

    PrintArray3(input.GetDataXRef());

    FaceCenteredGrid3D output;
    solver.Solve(input, sdf, sdfCollider, 0.5, 0.1, &output);

    PrintArray3(output.GetDataXRef());
}
