#include <gtest/gtest.h>
#include "../src/linear_system./blas.hpp"
#include "../src/linear_system./linear_system.hpp"

TEST(BLASTest, Dot_test)
{
    Vector3<double> v(1, 2, 3);
    Vector3<double> v1(1, 2, 3);
    EXPECT_EQ(14, BLAS::Dot(v, v1));
}

TEST(BLASTest, Residual_test)
{
    double a = 3;
    Vector3<double> x(1, 2, 3);
    Vector3<double> y(1, 2, 3);
    Vector3<double> result;
    BLAS::Residual(a, x, y, &result);
    EXPECT_EQ(true, result.IsEqual(Vector3<double>(2, 4, 6)));
}

TEST(BLASTest, AXpY_test)
{
    double a = 3;
    Vector3<double> x(1, 2, 3);
    Vector3<double> y(1, 2, 3);
    Vector3<double> result;
    BLAS::AXpY(a, x, y, &result);
    EXPECT_EQ(true, result.IsEqual(Vector3<double>(4, 8, 12)));
}

TEST(BLASTest, L2Norm_test)
{
    Vector3<double> x(1, 2, 3);
    EXPECT_EQ(sqrt(14), BLAS::L2Norm(x));
}

TEST(BLASTest, LInfNorm_test)
{
    Vector3<double> x(1, 2, -3);
    EXPECT_EQ(std::fabs(3), BLAS::LInfNorm(x));
}