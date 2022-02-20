#include <gtest/gtest.h>
#include "../src/solvers/forward_euler_diffusion_solver.hpp"

void AssertArray(Array3<double> expected, Array3<double> equal)
{
    EXPECT_EQ(true, equal.IsEqual(expected)); 
}

TEST(ForwardEulerDiffusionSolverTest, CalculateDiffusion_test)
{
    Array3<Vector3<double>> input_arr(2, 2, 2, 1);
    Array3<double> expected_output(2, 2, 2, 1);
    ForwardEulerDiffusionSolver solver = ForwardEulerDiffusionSolver();
    EXPECT_EQ(true, expected_output.IsEqual(solver.CalculateDiffusion(input_arr, 1, 1, 1, 1)));
}
