#include <gtest/gtest.h>
#include "../src/common/math_utils.hpp"
#include "../src/common/cuda_array_utils.hpp"
#include "../src/common/array3.hpp"


TEST(CudaArrayUtils, CUDA_ExtrapolateToRegion)
{
    const Vector3<size_t> size(20, 20, 20);
    Array3<int> markers(size, 0);
    Array3<double> vel(size, 0);
    Array3<double> vel_2(size, 0);

    vel(0, 0, 0) = 2;
    vel(1, 1, 0) = 8;
    vel(1, 0, 0) = 4;
    vel_2(0, 0, 0) = 2;
    vel_2(1, 1, 0) = 8;
    vel_2(1, 0, 0) = 4;
    
    markers(0, 0, 0) = 1;
    markers(1, 1, 0) = 1;
    markers(1, 0, 0) = 1;

    size_t iterations = 5;
    auto prevVel(vel);
    ExtrapolateToRegion(prevVel, markers, iterations, vel);
    WrappedCuda_ExtrapolateToRegion(prevVel, markers, iterations, vel_2);
    EXPECT_EQ(true, vel_2.IsEqual(vel));
}