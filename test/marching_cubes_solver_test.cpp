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
    TriangleMesh mesh;

    sdf(1, 1, 1) = -1;
    sdf(2, 1, 1) = -1;
    sdf(1, 2, 1) = -1;
    sdf(2, 2, 1) = -1;
    sdf(1, 1, 2) = -1;
    sdf(2, 1, 2) = -1;
    sdf(1, 2, 2) = -1;
    sdf(2, 2, 2) = -1;

    solver.BuildSurface(sdf, &mesh);

    objManager.Save("../../test_marching_cubes.obj", mesh);
}