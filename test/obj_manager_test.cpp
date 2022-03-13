#include <gtest/gtest.h>
#include "../src/3d/obj_manager.hpp"


TEST(OBJManagerTest, LoadSave_Test)
{
    OBJManager objManager;
    TriangleMesh mesh;
    objManager.Load("../../../test/test_cases/test_model.obj", &mesh);
    objManager.Save("../../saved_model.obj", mesh);
}