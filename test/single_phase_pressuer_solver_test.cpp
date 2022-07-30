#include <gtest/gtest.h>
#include "../src/fluid_solvers/single_phase_pressure_solver.hpp"

void ___PrintArray3(const Array3<double>& input)
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

TEST(SinglePhasePressureSolverTest, Solve_test)
{
    const Vector3<size_t> size(6, 6, 3);
    FaceCenteredGrid3D input(size, 0, 1, 1);
    ScalarGrid3D sdf(size, 1, Vector3<double>(0, 0, 0));
    ScalarGrid3D colliderSdf(size, 1, Vector3<double>(0, 0, 0));

    SinglePhasePressureSolver solver;

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

    //___PrintArray3(input.GetDataXRef());

    FaceCenteredGrid3D output;
    solver.Solve(input, sdf, colliderSdf, 1, 0.1, &output);

    //___PrintArray3(output.GetDataXRef());
//     __PrintArray3(output.GetDataYRef());
//     __PrintArray3(output.GetDataZRef());
}

TEST(SinglePhasePressureSolverTest, SolveSinglePhase) 
{
    std::cout << "Solve Single Phase\n";
    Vector3<size_t> size(4, 4, 4);
    FaceCenteredGrid3D vel(size);
    ScalarGrid3D sdf(size, -1, Vector3<double>(0, 0, 0));
    ScalarGrid3D colliderSdf(size, 1, Vector3<double>(0, 0, 0));
    vel.Fill(0, 0, 0);

    for (size_t k = 0; k < 4; ++k) 
    {
        for (size_t j = 0; j < 4; ++j) 
        {
            for (size_t i = 0; i < 4; ++i) 
            {
                if (j == 0 || j == 3) 
                {
                    vel.y(i, j, k) = 0.0;
                } 
                else 
                {
                    vel.y(i, j, k) = 1.0;
                }
            }
        }
    }

    SinglePhasePressureSolver solver;
    solver.Solve(vel, sdf, colliderSdf, 1, 1, &vel);

    for (size_t k = 0; k < 3; ++k) 
    {
        for (size_t j = 0; j < 3; ++j) 
        {
            for (size_t i = 0; i < 4; ++i) 
            {
                EXPECT_NEAR(0.0, vel.x(i, j, k), 1e-6);
            }
        }
    }

    for (size_t k = 0; k < 3; ++k) 
    {
        for (size_t j = 0; j < 4; ++j) 
        {
            for (size_t i = 0; i < 3; ++i) 
            {
                EXPECT_NEAR(0.0, vel.y(i, j, k), 1e-6);
            }
        }
    }

    for (size_t k = 0; k < 4; ++k) 
    {
        for (size_t j = 0; j < 3; ++j) 
        {
            for (size_t i = 0; i < 3; ++i) 
            {
                EXPECT_NEAR(0.0, vel.z(i, j, k), 1e-6);
            }
        }
    }

    const auto& pressure = solver.GetPressure();
    for (size_t k = 0; k < 3; ++k) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                EXPECT_NEAR(pressure(i, j + 1, k) - pressure(i, j, k), -1.0,
                            1e-6);
            }
        }
    }
}
