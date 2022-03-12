#include <gtest/gtest.h>
#include "../src/common/math_utils.hpp"


TEST(MathUtilsTest, Lerp_Test)
{
    double val = Lerp<double, double>(0, 2, 0.5);
    EXPECT_EQ(1, val);
}

TEST(MathUtilsTest, LerpVector3_Test)
{
    Vector3<double> v1(0, 1, 2);
    Vector3<double> v2(2, 3, 3);
    Vector3<double> val = Lerp<Vector3<double>, double>(v1, v2, 0.5);
    EXPECT_EQ(1, val.x);
    EXPECT_EQ(2, val.y);
    EXPECT_EQ(2.5, val.z);
}

TEST(MathUtilsTest, Bilerp_Test)
{
    double val = Bilerp<double, double>(0, 2, 0, 2, 0.5, 0.5);
    EXPECT_EQ(1, val);
}

TEST(MathUtilsTest, Trilerp_Test)
{
    double val = Trilerp<double, double>(0, 2, 0, 2, 0, 2, 0, 2, 0.5, 0.5, 0.5);
    EXPECT_EQ(1, val);
}