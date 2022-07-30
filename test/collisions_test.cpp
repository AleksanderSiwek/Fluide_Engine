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

#include <iostream>
TEST(CollisionsTest, ClosestTriangleIdx_test)
{
    Vector3<double> point0(0, 0.1, 0);
    Vector3<double> point1(0.5, 0.5, 0.5);
    Vector3<double> point2(0.99, 0.99, 0.99);
    Vector3<double> point3(1, 1, 1);
    Vector3<double> point4(1, 0, 1);
    Vector3<double> point5(1.5, 0, 0);
    TriangleMesh mesh;
    OBJManager objManager;

    objManager.Load("../../../test/test_cases/test_model.obj", &mesh);

    Vector3<double> position = point0;
    Vector3<double> velocity(0, -1, 0);

    double radius = 0;
    double restitutionCoefficient = 0;

    double frictionCoeffient = 0; // TO DO
    Vector3<double> colliderVelocity = 0; // TO DO

    const auto& triangles = mesh.GetTriangles();
    const auto& verts = mesh.GetVerticies();
    const auto& normals = mesh.GetNormals();
    size_t triangleIdx = Collisions::ClosestTriangleIdx(position, mesh);
    Vector3<double> closestPoint = Collisions::ClossestPointOnTriangle(position, verts[triangles[triangleIdx].point1Idx], verts[triangles[triangleIdx].point2Idx], verts[triangles[triangleIdx].point3Idx]);
    double distanceToTriangle = Collisions::DistanceToTriangle(position, verts[triangles[triangleIdx].point1Idx], verts[triangles[triangleIdx].point2Idx], verts[triangles[triangleIdx].point3Idx]);
    Vector3<double> closestNormal = normals[triangles[triangleIdx].normalIdx];

    if(mesh.IsInside(position))
    {
        Vector3<double> u = velocity.Dot(closestNormal) / (closestNormal.Dot(closestNormal)) * closestNormal;
        Vector3<double> w = velocity - u;
        velocity = w - u;
        // Vector3<double> relativeVel = velocity - colliderVelocity;
        // double normalDotRelativeVel = closestNormal.Dot(relativeVel);
        // Vector3<double> relativeVelN = normalDotRelativeVel * closestNormal;
        // Vector3<double> relativeVelT = relativeVel - relativeVelN;

        // if(normalDotRelativeVel < 0.0)
        // {
        //     Vector3<double> deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN;
        //     relativeVelN *= -restitutionCoefficient;

        //     if(relativeVelT.GetLengthSquared() > 0.0)
        //     {
        //         double frictionScale =  std::max(
        //             1.0 - frictionCoeffient * deltaRelativeVelN.GetLength() / relativeVelT.GetLength(),
        //             0.0);
        //             relativeVelT *= frictionScale;
        //     }
        // }
        //     velocity = relativeVelN + relativeVelT + colliderVelocity;
    }
    position = closestPoint;   

    // const auto& triangles = mesh.GetTriangles();
    // const auto& verts = mesh.GetVerticies();
    // size_t closestTriangleIdx = Collisions::ClosestTriangleIdx(point5, mesh);
    // double distance = Collisions::DistanceToTriangle(point5, 
    //                                 verts[triangles[closestTriangleIdx].point1Idx], 
    //                                 verts[triangles[closestTriangleIdx].point2Idx], 
    //                                 verts[triangles[closestTriangleIdx].point3Idx]);


    std::cout << "position: " << position.x << ", " << position.y << " " << position.z << "\n";
    std::cout << "velocity: " << velocity.x << ", " << velocity.y << " " << velocity.z << "\n";
    std::cout << "normal: " << closestNormal.x << ", " << closestNormal.y << " " << closestNormal.z << "\n";

    EXPECT_EQ(true, false);
    EXPECT_EQ(true, mesh.IsInside(point1));
    EXPECT_EQ(true, mesh.IsInside(point2));
    EXPECT_EQ(false, mesh.IsInside(point3));
    EXPECT_EQ(false, mesh.IsInside(point4));
    EXPECT_EQ(false, mesh.IsInside(point5));
}


