#include <gtest/gtest.h>
#include "../src/grid_systems/face_centered_grid3d.hpp"

TEST(FaceCenteredGrid3DTest, DefaultConstructor_test)
{
    FaceCenteredGrid3D grid;
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
    EXPECT_EQ(0, grid.x(0, 0, 0));
    EXPECT_EQ(0, grid.y(0, 0, 0));
    EXPECT_EQ(0, grid.z(0, 0, 0));
}

TEST(FaceCenteredGrid3DTest, Constructor_test)
{
    FaceCenteredGrid3D grid(2, 1, Vector3<double>(1, 2, 3), 1);
    auto size = grid.GetSize();
    auto actualSize = grid.GetActualSize();
    auto origin = grid.GetOrigin();
    auto spacing = grid.GetGridSpacing();
    EXPECT_EQ(2, size.x);
    EXPECT_EQ(2, size.y);
    EXPECT_EQ(2, size.z);
    EXPECT_EQ(3, actualSize.x);
    EXPECT_EQ(3, actualSize.y);
    EXPECT_EQ(3, actualSize.z);
    EXPECT_EQ(1, origin.x);
    EXPECT_EQ(1, origin.y);
    EXPECT_EQ(1, origin.z);
    EXPECT_EQ(1, spacing.x);
    EXPECT_EQ(2, spacing.y);
    EXPECT_EQ(3, spacing.z);
    EXPECT_EQ(1, grid.x(2, 2, 2));
    EXPECT_EQ(1, grid.y(2, 2, 2));
    EXPECT_EQ(1, grid.z(2, 2, 2));
}

TEST(FaceCenteredGrid3DTest, CalculationOrigin_test)
{
    FaceCenteredGrid3D grid(2, 0, 1, 1);
    EXPECT_EQ(0, grid.GetDataXOrigin().x);
    EXPECT_EQ(0.5, grid.GetDataXOrigin().y);
    EXPECT_EQ(0.5, grid.GetDataXOrigin().z);
}

TEST(FaceCenteredGrid3DTest, ValueAtCellCenter_test)
{
    FaceCenteredGrid3D grid(3, 0, 1, 3);
    Vector3<double> valueAtCenter = grid.ValueAtCellCenter(1, 1, 1);
    EXPECT_EQ(3, valueAtCenter.x);
    EXPECT_EQ(3, valueAtCenter.y);
    EXPECT_EQ(3, valueAtCenter.z);
}