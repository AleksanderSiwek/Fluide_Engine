#include <gtest/gtest.h>
#include "../src/linear_system./cuda_blas.hpp"
#include "../src/linear_system/linear_system.hpp"

#include <iostream>

TEST(CUDA_BLASTest, Dot_test)
{
    SystemVector v1(2, 2, 2, 2);
    SystemVector v2(2, 2, 2, 3);

    double ret = CUDA_BLAS::Dot(v1, v2);
    EXPECT_EQ(48, ret);
}

TEST(CUDA_BLASTest, Residual_test)
{
    struct LinearSystemMatrixRow initialValue = {2, 3, 5, 4};
    SystemMatrix A(2, 2, 2, initialValue);
    SystemVector v1(2, 2, 2, 2);
    SystemVector v2(2, 2, 2, 3);
    SystemVector result(2, 2, 2, 3);
    CUDA_BLAS::Residual(A, v1, v2, &result);
    SystemVector expected(2, 2, 2, -25);
    EXPECT_EQ(true, result.IsEqual(expected));
}

TEST(CUDA_BLASTest, L2Norm_test)
{
    SystemVector x(2, 2, 2, 2);
    EXPECT_EQ(sqrt(32), CUDA_BLAS::L2Norm(x));
}
