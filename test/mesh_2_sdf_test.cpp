#include <gtest/gtest.h>
#include "../src/3d/mesh_2_sdf.hpp"
#include "../src/3d/obj_manager.hpp"


TEST(MeshSignedDistanceFieldTest, DistanceField_Test)
{
    OBJManager objManager;
    TriangleMesh mesh;
    objManager.Load("../../../test/test_cases/test_cube.obj", &mesh);
    ScalarGrid3D sdf(Vector3<size_t>(3, 3, 3));
    Mesh2SDF sdfBuilder;
    sdfBuilder.Build(mesh, sdf);
    // const auto& triangles = mesh.GetTriangles();
    // Vector3<double> point1(1, 1, 0);
    // Vector3<double> point2(3, 1, 0);
    // Vector3<double> point3(3, 3, 0);
    for(size_t i = 0; i < sdf.GetSize().x; i++)
    {
        for(size_t j = 0; j < sdf.GetSize().y; j++)
        {
            for(size_t k = 0; k < sdf.GetSize().z; k++)
            {
                auto pos = sdf.GridIndexToPosition(i, j, k);
                //double distance = Collisions::DistanceToTriangle(pos, point1, point2, point3);
                //Vector3<double> closestPoint = Collisions::ClossestPointOnTriangle(pos, point1, point2, point3);
                std::cout << "GridIdx to position: " << pos.x << ", " << pos.y << ", " << pos.z << "\n";
                //std::cout << "Closest Point : " << closestPoint.x << ", " << closestPoint.y << ", " << closestPoint.z << "\n"; 
                std::cout << "Distance(" << i << ", " << j << ", " << k <<"): " << sdf(i, j, k) << "\n\n";
            }
        }
    }
}