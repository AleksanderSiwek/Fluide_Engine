#include <gtest/gtest.h>
#include "../src/3d/bounding_box_3d.hpp"


TEST(BoundingBox3DTest, DefaultConstructor_test)
{
    BoundingBox3D box = BoundingBox3D();
    EXPECT_EQ(0, box.GetOrigin().x);
    EXPECT_EQ(0, box.GetOrigin().y);
    EXPECT_EQ(0, box.GetOrigin().z);
    EXPECT_EQ(1, box.GetSize().x);
    EXPECT_EQ(1, box.GetSize().y);
    EXPECT_EQ(1, box.GetSize().z);
}

TEST(BoundingBox3DTest, Constructor_test)
{
    BoundingBox3D box = BoundingBox3D(1, 3);
    EXPECT_EQ(1, box.GetOrigin().x);
    EXPECT_EQ(1, box.GetOrigin().y);
    EXPECT_EQ(1, box.GetOrigin().z);
    EXPECT_EQ(3, box.GetSize().x);
    EXPECT_EQ(3, box.GetSize().y);
    EXPECT_EQ(3, box.GetSize().z);
}

TEST(BoundingBox3DTest, IsInside_test)
{
    BoundingBox3D box = BoundingBox3D(0, 3);
    Vector3<double> position_1(0, 1, 1);
    Vector3<double> position_2(0, 1, 3.01);
    EXPECT_EQ(true, box.IsInside(position_1));
    EXPECT_EQ(false, box.IsInside(position_2));
}