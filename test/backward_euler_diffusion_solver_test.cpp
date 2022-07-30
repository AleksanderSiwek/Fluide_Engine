#include <gtest/gtest.h>
#include "../src/fluid_solvers/backward_euler_diffusion_solver.hpp"
#include "../src/fluid_solvers/forward_euler_diffusion_solver.hpp"
#include <iomanip>


void _PrintArray3(const Array3<double>& input)
{
    const auto& size = input.GetSize();
    std::cout << std::setprecision(3) << std::fixed;
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

TEST(BackwardEulerDiffusionSolverTest, BuilSystem_test)
{
    // const Vector3<size_t> size(3, 3, 1);
    // FaceCenteredGrid3D input(size, 0, 1, 1);
    // ScalarGrid3D sdf(size, 1, Vector3<double>(0, 0, 0));
    // BackwardEulerDiffusionSolver solver;

    // auto& xData = input.GetDataXRef();
    // auto& yData = input.GetDataYRef();
    // auto& zData = input.GetDataZRef();

    // xData(1, 0, 0) = yData(1, 1, 0) = zData(1, 1, 0) = 5;
    // xData(1, 1, 0) = yData(0, 1, 0) = zData(0, 1, 0) = 5;
    // sdf(1, 0, 0) = -1;
    // sdf(1, 1, 0) = -1;

    // _PrintArray3(input.GetDataXRef());
    // std::cout << std::setprecision(2);
    // FaceCenteredGrid3D output;
    // solver.Solve(input, sdf, 0.5, 0.1, &output);
    // std::cout << "Backward Euler result:\n";
    // _PrintArray3(output.GetDataXRef());

    // ForwardEulerDiffusionSolver forwardSolver;
    // FaceCenteredGrid3D output1;
    // forwardSolver.Solve(input, sdf, 0.5, 0.1, &output1);
    // std::cout << "Forward Euler result:\n";
    // _PrintArray3(output1.GetDataXRef());
}

TEST(BackwardEulerDiffusionSolverTest, Solve_test)
{
    const Vector3<size_t> size(6, 6, 3);
    FaceCenteredGrid3D input(size, 0, 1, 1);
    ScalarGrid3D sdf(size, 1, Vector3<double>(0, 0, 0));
    ScalarGrid3D sdfCollider(size, 1, Vector3<double>(0, 0, 0));
    BackwardEulerDiffusionSolver solver;

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

    _PrintArray3(input.GetDataXRef());

    FaceCenteredGrid3D output;
    solver.Solve(input, sdf, sdfCollider, 0.5, 0.1, &output);

    _PrintArray3(output.GetDataXRef());
}

#include "../src/fluid_solvers/single_phase_pressure_solver.hpp"
TEST(BackwardEulerDiffusionSolverTest, Solve2_test)
{
    const Vector3<size_t> size(5, 5, 5);
    FaceCenteredGrid3D input(size, 0, 1, 0);
    ScalarGrid3D sdf(size, 0, Vector3<double>(0.5, 0.5, 0.5));
    ScalarGrid3D sdfCollider(size, 1, Vector3<double>(0, 0, 0));
    BackwardEulerDiffusionSolver solver;

    auto& xData = input.GetDataXRef();
    auto& yData = input.GetDataYRef();
    auto& zData = input.GetDataZRef();

    xData(0, 0, 0) = yData(0, 0, 0) = zData(0, 0, 0) = 5;
    xData(1, 0, 0) = yData(1, 0, 0) = zData(1, 0, 0) = 5;
    xData(0, 1, 0) = yData(0, 1, 0) = zData(0, 1, 0) = 5;
    xData(1, 1, 0) = yData(1, 1, 0) = zData(1, 1, 0) = 10;
    xData(0, 0, 1) = yData(0, 0, 1) = zData(0, 0, 1) = 10;
    xData(1, 0, 1) = yData(1, 0, 1) = zData(1, 0, 1) = 2;
    xData(0, 1, 1) = yData(0, 1, 1) = zData(0, 1, 1) = 2;
    xData(1, 1, 1) = yData(1, 1, 1) = zData(1, 1, 1) = 2;

    sdf.Fill(5);
    sdf(0, 0, 0) = -5;
    sdf(1, 0, 0) = -5;
    sdf(0, 1, 0) = -5;
    sdf(1, 1, 0) = -5;
    sdf(0, 0, 1) = -5;
    sdf(1, 0, 1) = -5;
    sdf(0, 1, 1) = -5;
    sdf(1, 1, 1) = -5;

    std::cout << "GetXPos(1, 1, 1): " << input.GetXPos(1, 1, 1).x << ", " << input.GetXPos(1, 1, 1).y << ", " << input.GetXPos(1, 1, 1).z << "\n" ;
    std::cout << "Is Inside: " << sdf.Sample(input.GetXPos(1, 1, 1)) << "\n";
    std::cout << "GetXPos(2, 2, 1): " << input.GetXPos(2, 2, 1).x << ", " << input.GetXPos(2, 2, 1).y << ", " << input.GetXPos(2, 2, 1).z << "\n" ;
    std::cout << "Is Inside: " << sdf.Sample(input.GetXPos(2, 2, 1)) << "\n";

    auto div = input.DivergenceAtCallCenter(0, 0, 0);
    

    FaceCenteredGrid3D output;
    solver.Solve(input, sdf, sdfCollider, 1, 0.05, &output);

    _PrintArray3(output.GetDataXRef());
    _PrintArray3(output.GetDataZRef());
    _PrintArray3(output.GetDataYRef());
}