#include <gtest/gtest.h>
#include "../src/3d/obj_manager.hpp"


TEST(OBJManagerTest, LoadSave_Test)
{
    OBJManager objManager;
    Mesh mesh;
    objManager.Load("../../../test/test_cases/test_model.obj", &mesh);
    mesh.AddVertex(Vector3<double>(1, 1, 2));
    mesh.AddVertex(Vector3<double>(-1, 1, 2));
    mesh.AddNormal({0.0, 0.0, 1});
    std::vector<size_t> face({3, 7, 10, 9});
    mesh.AddFace(face);
    objManager.Save("../../saved_model.obj", mesh);
}