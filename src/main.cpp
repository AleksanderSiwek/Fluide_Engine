#include "pic_simulator.hpp"
#include "3d/obj_manager.hpp"
#include "forces/directional_field.hpp"
#include "forces/point_field.hpp"
#include "3d/collider_collection.hpp"
#include "3d/triangle_mesh_collider.hpp"

#include <stdlib.h>  
#include <crtdbg.h>   //for malloc and free
#include <iostream>

int main()
{
    size_t resolution = 40;
    Vector3<int> scalers(1, 2, 2);
    size_t numberOfIterations = 150;
    double timeIntervalInSeconds = 0.04;

    const Vector3<size_t> size(resolution*scalers.x, resolution*scalers.y, resolution*scalers.z);
    Vector3<double> domainOrigin(0, 0, 0);
    Vector3<double> domainSize(4*scalers.x, 4*scalers.y, 4*scalers.z);
    BoundingBox3D domain(domainOrigin, domainSize);

    // Load fluid mesh
    TriangleMesh fluidMesh;
    OBJManager objLoader;
    objLoader.Load("../../../test/test_cases/2_Walls.obj", &fluidMesh);

    // Setup colliders
    TriangleMesh colliderMesh_1;
    TriangleMesh colliderMesh_2;
    objLoader.Load("../../../test/test_cases/collider_1.obj", &colliderMesh_1);
    objLoader.Load("../../../test/test_cases/test_cube.obj", &colliderMesh_2);
    auto collider_1 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide(Vector3<double>((double)size.x, (double)size.y, (double)size.z)), colliderMesh_1);
    auto collider_2 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide(Vector3<double>((double)size.x, (double)size.y, (double)size.z)), colliderMesh_2);

    // Setup Simulator
    PICSimulator simulator(size, domain);
    simulator.AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    //simulator.AddExternalForce(std::make_shared<PointField>(Vector3<double>(2, 2, 2), 10));
    simulator.InitializeFrom3dMesh(fluidMesh);
    simulator.SetViscosity(0);
    simulator.AddCollider(collider_1);
    //simulator.AddCollider(collider_2);
    simulator.SetMaxClf(1);

    TriangleMesh tmpMesh;

    Frame frame(timeIntervalInSeconds);
    simulator.SetCurrentFrame(frame);
    for(size_t i = 0; i < numberOfIterations; i++)
    {
        std::cout << "Iteration = " << i << "\n";
        simulator.AdvanceSingleFrame();
        simulator.GetSurface(&tmpMesh);
        objLoader.Save("../../simulation_test_" + std::to_string(i) + ".obj", tmpMesh);
        tmpMesh.Clear();
    }   
}