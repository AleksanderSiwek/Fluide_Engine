#include <gtest/gtest.h>
#include "../src/3d/scalar_grid3d.hpp"

TEST(ScalarGrid3DTest, DefaultConstructor_test)
{
    ScalarGrid3D grid;
    auto size = grid.GetSize();
    auto origin = grid.GetOrigin();
    auto spacing = grid.GetGridSpacing();
    EXPECT_EQ(1, size.x);
    EXPECT_EQ(1, size.y);
    EXPECT_EQ(1, size.z);
    EXPECT_EQ(0, origin.x);
    EXPECT_EQ(0, origin.y);
    EXPECT_EQ(0, origin.z);
    EXPECT_EQ(1, spacing.x);
    EXPECT_EQ(1, spacing.y);
    EXPECT_EQ(1, spacing.z);
    EXPECT_EQ(0, grid(0, 0, 0));
}

TEST(ScalarGrid3DTest, Constructor_test)
{
    ScalarGrid3D grid(2, 1, Vector3<double>(1, 2, 3), 1);
    auto size = grid.GetSize();
    auto origin = grid.GetOrigin();
    auto spacing = grid.GetGridSpacing();
    EXPECT_EQ(2, size.x);
    EXPECT_EQ(2, size.y);
    EXPECT_EQ(2, size.z);
    EXPECT_EQ(1, origin.x);
    EXPECT_EQ(1, origin.y);
    EXPECT_EQ(1, origin.z);
    EXPECT_EQ(1, spacing.x);
    EXPECT_EQ(2, spacing.y);
    EXPECT_EQ(3, spacing.z);
    EXPECT_EQ(1, grid(1, 1, 1));
}

TEST(ScalarGrid3DTest, Gradient_test)
{
    ScalarGrid3D grid(2, 1, (1, 2, 3), 1);
    Vector3<double> pos(1, 1.2, 0.5);
    Vector3<double> grad = grid.Gradient(pos);
    EXPECT_EQ(1, grad.x);
    EXPECT_EQ(1, grad.y);
    EXPECT_EQ(1, grad.z);
}

TEST(ScalarGrid3DTest, Laplacian_test)
{
    ScalarGrid3D grid(2, 1, (1, 2, 3), 1);
    Vector3<double> pos(1, 1.2, 0.5);
    EXPECT_EQ(1, grid.Laplacian(pos));
}