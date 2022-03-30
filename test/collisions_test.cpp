#include <gtest/gtest.h>
#include "../src/3d/collisions.hpp"

TEST(CollisionsTest, DistanceToPoint_test)
{
    Vector3<double> p1(0, 1, 0);
    Vector3<double> p2(0, 0, 0);
    double distance = Collisions::DistanceToPoint(p1, p2);
    EXPECT_EQ(1, distance);
}

TEST(CollisionsTest, ClossestPointOnTriangle_test)
{
    Vector3<double> point(0, 1, 0.5);
    Vector3<double> triP1(1, 0, 0);
    Vector3<double> triP2(-1, 0, 0);
    Vector3<double> triP3(0, 0, 1);

    Vector3<double> pointOnTriangle = Collisions::ClossestPointOnTriangle(point, triP1, triP2, triP3);

    EXPECT_EQ(0, pointOnTriangle.x);
    EXPECT_EQ(0, pointOnTriangle.y);
    EXPECT_EQ(0.5, pointOnTriangle.z);
}

TEST(CollisionsTest, DistanceToTriangle_test)
{
    Vector3<double> point(0, 1, 0.5);
    Vector3<double> triP1(1, 0, 0);
    Vector3<double> triP2(-1, 0, 0);
    Vector3<double> triP3(0, 0, 1);

    double distance = Collisions::DistanceToTriangle(point, triP1, triP2, triP3);

    EXPECT_EQ(1, distance);
}

TEST(CollisionsTest, DistanceToTriangle2_test)
{
    Vector3<double> point(0, 0, 2);
    Vector3<double> triP1(1, 0, 0);
    Vector3<double> triP2(-1, 0, 0);
    Vector3<double> triP3(0, 0, 1);

    auto p = Collisions::ClossestPointOnTriangle(point, triP1, triP2, triP3);
    std::cout << "Point: " << p.x << ", " << p.y << ", " << p.z << "\n";

    double distance = Collisions::DistanceToTriangle(point, triP1, triP2, triP3);

    EXPECT_EQ(1, distance);
}


