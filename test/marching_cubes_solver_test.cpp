#include <gtest/gtest.h>
#include "../src/3d/marching_cubes_solver.hpp"
#include "../src/3d/obj_manager.hpp"
#include "../src/3d/scalar_grid3d.hpp"


TEST(MarchingCubesSolverTest, BuildMesh_test)
{
    Vector3<size_t> size(5, 5, 5);
    Vector3<double> origin(0, 0, 0);
    Vector3<double> gridSpacing(0.5, 0.5, 0.5);

    MarchingCubesSolver solver;
    OBJManager objManager;
    ScalarGrid3D sdf(size, 1, origin, gridSpacing);
    ScalarGrid3D sdfColl(size, 1, origin, gridSpacing);
    TriangleMesh mesh;

    sdf(0, 1, 1) = -1;
    sdf(0, 1, 0) = -1;
    sdf(0, 0, 1) = -1;
    sdf(0, 0, 0) = -1;
    sdf(1, 1, 1) = -1;
    sdf(1, 1, 0) = -1;
    sdf(1, 0, 1) = -1;
    sdf(1, 0, 0) = -1;

    solver.BuildSurface(sdf, sdfColl, mesh);

    objManager.Save("../../test_marching_cubes.obj", mesh);
}