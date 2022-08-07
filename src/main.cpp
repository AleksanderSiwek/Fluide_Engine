#include "pic_simulator.hpp"
#include "flip_simulator.hpp"
#include "apic_simulator.hpp"
#include "3d/obj_manager.hpp"
#include "forces/directional_field.hpp"
#include "forces/point_field.hpp"
#include "3d/collider_collection.hpp"
#include "3d/triangle_mesh_collider.hpp"

#include <stdlib.h>  
#include <iostream>

int main()
{
    size_t resolution = 30;
    Vector3<double> scalers(1, 1.5, 2);
    size_t numberOfIterations = 240;
    double timeIntervalInSeconds = 1.0 / 60.0;

    const Vector3<size_t> size((size_t)(resolution*scalers.x), (size_t)(resolution*scalers.y), (size_t)(resolution*scalers.z));
    Vector3<double> domainOrigin(0, 0, 0);
    Vector3<double> domainSize(4*scalers.x, 4*scalers.y, 4*scalers.z);
    BoundingBox3D domain(domainOrigin, domainSize);

    // Load fluid mesh
    TriangleMesh fluidMesh;
    OBJManager objLoader;
    objLoader.Load("../../../test/test_cases/water_wall.obj", &fluidMesh);

    // Setup colliders
    TriangleMesh colliderMesh_1;
    TriangleMesh colliderMesh_2;
    objLoader.Load("../../../test/test_cases/collider_2.obj", &colliderMesh_1);
    objLoader.Load("../../../test/test_cases/test_cube.obj", &colliderMesh_2);
    auto collider_1 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide(Vector3<double>((double)size.x, (double)size.y, (double)size.z)), colliderMesh_1);
    auto collider_2 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide(Vector3<double>((double)size.x, (double)size.y, (double)size.z)), colliderMesh_2);

    // Setup PIC Simulator
    PICSimulator picSimulator(size, domain);

    picSimulator.AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    //simulator.AddExternalForce(std::make_shared<PointField>(Vector3<double>(2, 2, 2), 10));
    picSimulator.InitializeFrom3dMesh(fluidMesh);
    picSimulator.SetViscosity(0);
    picSimulator.AddCollider(collider_1);
    //simulator.AddCollider(collider_2);
    picSimulator.SetMaxClf(3);

    TriangleMesh tmpMesh;

    Frame frame(timeIntervalInSeconds);
    picSimulator.SetCurrentFrame(frame);
    // for(size_t i = 0; i < numberOfIterations; i++)
    // {
    //     std::cout << "Iteration = " << i << "\n";
    //     picSimulator.AdvanceSingleFrame();
    //     picSimulator.GetSurface(&tmpMesh);
    //     objLoader.Save("../../pic_simulation_test_" + std::to_string(i) + ".obj", tmpMesh);
    //     tmpMesh.Clear();
    // }   

    // Setup APIC Simulator
    APICSimulator apicSimulator(size, domain);
    apicSimulator.AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    //simulator.AddExternalForce(std::make_shared<PointField>(Vector3<double>(2, 2, 2), 10));
    apicSimulator.InitializeFrom3dMesh(fluidMesh);
    apicSimulator.SetViscosity(0);
    apicSimulator.AddCollider(collider_1);
    //simulator.AddCollider(collider_2);
    apicSimulator.SetMaxClf(3);

    Frame apicFrame(timeIntervalInSeconds);
    apicSimulator.SetCurrentFrame(apicFrame);
    for(size_t i = 0; i < numberOfIterations; i++)
    {
        std::cout << "Iteration = " << i << "\n";
        apicSimulator.AdvanceSingleFrame();
        apicSimulator.GetSurface(&tmpMesh);
        objLoader.Save("../../apic_simulation_test_" + std::to_string(i) + ".obj", tmpMesh);
        tmpMesh.Clear();
    }   

    // Setup FLIP Simulator
    FLIPSimulator flipSimulator(size, domain);
    flipSimulator.AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    //simulator.AddExternalForce(std::make_shared<PointField>(Vector3<double>(2, 2, 2), 10));
    flipSimulator.InitializeFrom3dMesh(fluidMesh);
    flipSimulator.SetViscosity(0);
    flipSimulator.AddCollider(collider_1);
    //simulator.AddCollider(collider_2);
    flipSimulator.SetMaxClf(3);

    Frame flipFrame(timeIntervalInSeconds);
    flipSimulator.SetCurrentFrame(flipFrame);
    for(size_t i = 0; i < numberOfIterations; i++)
    {
        std::cout << "Iteration = " << i << "\n";
        flipSimulator.AdvanceSingleFrame();
        flipSimulator.GetSurface(&tmpMesh);
        objLoader.Save("../../flip_simulation_test_" + std::to_string(i) + ".obj", tmpMesh);
        tmpMesh.Clear();
    }  
}