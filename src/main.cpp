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
    size_t dimSize = 40;
    size_t numberOfIterations = 100;
    double timeIntervalInSeconds = 0.05;

    const Vector3<size_t> size(dimSize, dimSize, dimSize*2);
    Vector3<double> domainOrigin(0, 0, 0);
    Vector3<double> domainSize(4, 4, 8);
    BoundingBox3D domain(domainOrigin, domainSize);

    // Load fluid mesh
    TriangleMesh fluidMesh;
    OBJManager objLoader;
    objLoader.Load("../../../test/test_cases/water_wall.obj", &fluidMesh);

    // Setup colliders
    TriangleMesh colliderMesh_1;
    TriangleMesh colliderMesh_2;
    objLoader.Load("../../../test/test_cases/collider_1.obj", &colliderMesh_1);
    //objLoader.Load("../../../test/test_cases/collider_2.obj", &colliderMesh_2);
    auto collider_1 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide((double)size.x), colliderMesh_1);
    //auto collider_2 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide((double)size.x), colliderMesh_2);

    // Setup Simulator
    PICSimulator simulator(size, domain);
    simulator.AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    //simulator.AddExternalForce(std::make_shared<PointField>(Vector3<double>(2, 2, 2), 10));
    simulator.InitializeFrom3dMesh(fluidMesh);
    simulator.SetViscosity(0);
    simulator.AddCollider(collider_1);
    simulator.SetMaxClf(1);
    //simulator.AddCollider(collider_2);

    TriangleMesh tmpMesh = colliderMesh_1;

    Frame frame(timeIntervalInSeconds);
    simulator.SetCurrentFrame(frame);
    for(size_t i = 0; i < numberOfIterations; i++)
    {
        std::cout << "Iteration = " << i << "\n";
        simulator.AdvanceSingleFrame();
        simulator.GetSurface(&tmpMesh);
        objLoader.Save("../../simulation_test_" + std::to_string(i) + ".obj", tmpMesh);
        tmpMesh.Clear();
        //tmpMesh = colliderMesh_1;
    }   
}