#include <gtest/gtest.h>
#include "../src/linear_system./blas.hpp"
#include "../src/linear_system./linear_system.hpp"

TEST(BLASTest, Dot_test)
{
    SystemVector v1(2, 2, 2, 2);
    SystemVector v2(2, 2, 2, 3);
    EXPECT_EQ(48, BLAS::Dot(v1, v2));
}

TEST(BLASTest, Residual_test)
{
    struct LinearSystemMatrixRow initialValue = {2, 3, 5, 4};
    SystemMatrix A(2, 2, 2, initialValue);
    SystemVector v1(2, 2, 2, 2);
    SystemVector v2(2, 2, 2, 3);
    SystemVector result(2, 2, 2, 3);
    BLAS::Residual(A, v1, v2, &result);
    SystemVector expected(2, 2, 2, -25);
    EXPECT_EQ(true, result.IsEqual(expected));
}

TEST(BLASTest, AXpY_test)
{
    double a = 3;
    SystemVector x(2, 2, 2, 2);
    SystemVector y(2, 2, 2, 3);
    SystemVector result(x.GetSize());
    BLAS::AXpY(a, x, y, &result);
    EXPECT_EQ(true, result.IsEqual(SystemVector(2, 2, 2, 9)));
}

TEST(BLASTest, L2Norm_test)
{
    SystemVector x(2, 2, 2, 2);
    EXPECT_EQ(sqrt(32), BLAS::L2Norm(x));
}

TEST(BLASTest, LInfNorm_test)
{
    SystemVector x(2, 2, 2, 1);
    x(0, 0, 0) = -5;
    x(1, 1, 1) = 4;
    EXPECT_EQ(std::fabs(5), BLAS::LInfNorm(x));
}