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

    Vector3<size_t> size = sdf.GetSize();
    for(int i = 0; i < size.x; i++)
    {
        for(int j = 0; j < size.y; j++)
        {
            for(int k = 0; k < size.z; k++)
            {
                auto pos = sdf.GridIndexToPosition(i, j, k);
                // std::cout << "sdf(" << i << ", " << j << ", " << k << ") = " << sdf(i, j, k) << "\n";
                // std::cout << "position = " << pos.x << ", " << pos.y << ", " << pos.z << "\n\n";
            }
        }
    }
}

