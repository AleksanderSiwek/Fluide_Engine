#include <gtest/gtest.h>
#include "../src/3d/collisions.hpp"
#include "../src/3d/obj_manager.hpp"

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
    double distance = Collisions::DistanceToTriangle(point, triP1, triP2, triP3);

    EXPECT_EQ(1, distance);
}

TEST(CollisionsTest, IsInsideTriangleMesh_test)
{
    Vector3<double> point0(0, 0, 0);
    Vector3<double> point1(0.5, 0.5, 0.5);
    Vector3<double> point2(0.99, 0.99, 0.99);
    Vector3<double> point3(1, 1, 1);
    Vector3<double> point4(1, 0, 1);
    Vector3<double> point5(2, 1, 1);
    TriangleMesh mesh;
    OBJManager objManager;

    objManager.Load("../../../test/test_cases/test_model.obj", &mesh);

    EXPECT_EQ(true, mesh.IsInside(point0));
    EXPECT_EQ(true, mesh.IsInside(point1));
    EXPECT_EQ(true, mesh.IsInside(point2));
    EXPECT_EQ(false, mesh.IsInside(point3));
    EXPECT_EQ(false, mesh.IsInside(point4));
    EXPECT_EQ(false, mesh.IsInside(point5));
}


