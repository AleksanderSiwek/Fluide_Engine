#include <gtest/gtest.h>
#include "../src/solvers/forward_euler_diffusion_solver.hpp"

#define AIR_CELL 0 
#define FLUID_CELL 1

TEST(ForwardEulerDiffusionSolverTest, CalculateDiffusion_test)
{
    FaceCenteredGrid3D input_arr(3, 0, 1, 1);
    Array3<uint8_t> fluid_markers(3, AIR_CELL);
    FaceCenteredGrid3D expected_output = FaceCenteredGrid3D(3, 0, 1, 0);
    fluid_markers(0, 0, 2) = FLUID_CELL;
    fluid_markers(0, 1, 2) = FLUID_CELL;
    fluid_markers(1, 0, 2) = FLUID_CELL;
    fluid_markers(1, 1, 2) = FLUID_CELL;
    ForwardEulerDiffusionSolver solver = ForwardEulerDiffusionSolver();
    FaceCenteredGrid3D output = solver.CalculateDiffusion(input_arr, fluid_markers, 0.5, 0.1);
    EXPECT_EQ(true, expected_output.IsEqual(output));
}
