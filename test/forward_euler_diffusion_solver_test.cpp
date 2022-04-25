#include <gtest/gtest.h>
#include "../src/fluid_solvers/forward_euler_diffusion_solver.hpp"
#include <iomanip>

void PrintArray3(const Array3<double>& input)
{
    const auto& size = input.GetSize();
    std::cout << std::setprecision(1) << std::fixed;
    std::cout << "Size: (" << size.x << ", " << size.y << ", " << size.z << ")\n";
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                std::cout << input(i, j, k) << " ";
            }
            std::cout << "      ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

TEST(ForwardEulerDiffusionSolverTest, CalculateDiffusion_test)
{
    const Vector3<size_t> size(6, 1, 6);
    FaceCenteredGrid3D input(size, 0, 1, 1);
    ScalarGrid3D sdf(size, 1);
    ForwardEulerDiffusionSolver solver;

    auto& xData = input.GetDataXRef();
    auto& yData = input.GetDataYRef();
    auto& zData = input.GetDataZRef();

    xData(2, 0, 2) = yData(2, 0, 2) = zData(2, 0, 2) = 2;
    xData(3, 0, 2) = yData(3, 0, 2) = zData(3, 0, 2) = 2;
    xData(2, 0, 3) = yData(2, 0, 3) = zData(2, 0, 3) = 2;
    xData(3, 0, 3) = yData(3, 0, 3) = zData(3, 0, 3) = 2;

    sdf(3, 0, 3) = -10;
    sdf(4, 0, 3) = -10;
    sdf(3, 0, 4) = -10;
    sdf(4, 0, 4) = -10;

    PrintArray3(input.GetDataXRef());

    FaceCenteredGrid3D output;
    solver.Solve(input, sdf, 0.5, 0.1, &output);

    PrintArray3(output.GetDataXRef());
}
