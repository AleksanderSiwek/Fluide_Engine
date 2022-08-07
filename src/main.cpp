#include "pic_simulator.hpp"
#include "cuda_pic_simulator.hpp"
#include "flip_simulator.hpp"
#include "apic_simulator.hpp"
#include "3d/obj_manager.hpp"
#include "forces/directional_field.hpp"
#include "forces/point_field.hpp"
#include "3d/collider_collection.hpp"
#include "3d/triangle_mesh_collider.hpp"

#include <stdlib.h>  
#include <iostream>

void RunSimulation(std::shared_ptr<PICSimulator> simulator, size_t numberOfIterations, double timeIntervalInSeconds, std::string prefix, Vector3<size_t> size, BoundingBox3D domain)
{
    // Load fluid mesh
    TriangleMesh fluidMesh;
    OBJManager objLoader;
    objLoader.Load("../../../test/test_cases/short_water_wall.obj", &fluidMesh);

    // Setup colliders
    TriangleMesh colliderMesh_1;
    TriangleMesh colliderMesh_2;
    TriangleMesh tmpMesh;
    objLoader.Load("../../../test/test_cases/collider_2.obj", &colliderMesh_1);
    objLoader.Load("../../../test/test_cases/test_cube.obj", &colliderMesh_2);
    auto domainOrigin = domain.GetOrigin();
    auto domainSize = domain.GetSize();
    auto collider_1 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide(Vector3<double>((double)size.x, (double)size.y, (double)size.z)), colliderMesh_1);
    auto collider_2 = std::make_shared<TriangleMeshCollider>(size, domainOrigin, (domainSize - domainOrigin).Divide(Vector3<double>((double)size.x, (double)size.y, (double)size.z)), colliderMesh_2);

    simulator->AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    //simulator.AddExternalForce(std::make_shared<PointField>(Vector3<double>(2, 2, 2), 10));
    simulator->InitializeFrom3dMesh(fluidMesh);
    simulator->SetViscosity(0);
    simulator->AddCollider(collider_1);
    //simulator.AddCollider(collider_2);
    simulator->SetMaxClf(3);

    Frame picFrame(timeIntervalInSeconds);
    simulator->SetCurrentFrame(picFrame);
    for(size_t i = 0; i < numberOfIterations; i++)
    {
        std::cout << "Iteration = " << i << "\n";
        simulator->AdvanceSingleFrame();
        simulator->GetSurface(&tmpMesh);
        objLoader.Save("../../" + prefix + "_" + std::to_string(i) + ".obj", tmpMesh);
        tmpMesh.Clear();
    }   

}

int main()
{
    size_t resolution = 75;
    Vector3<double> scalers(1, 1.5, 2);
    size_t numberOfIterations = 5;
    double timeIntervalInSeconds = 1.0 / 60.0;

    const Vector3<size_t> size((size_t)(resolution*scalers.x), (size_t)(resolution*scalers.y), (size_t)(resolution*scalers.z));
    Vector3<double> domainOrigin(0, 0, 0);
    Vector3<double> domainSize(4*scalers.x, 4*scalers.y, 4*scalers.z);
    BoundingBox3D domain(domainOrigin, domainSize);

    std::shared_ptr<PICSimulator> picSimulator = std::make_shared<PICSimulator>(size, domain);
    std::shared_ptr<APICSimulator> apicSimulator = std::make_shared<APICSimulator>(size, domain);
    std::shared_ptr<FLIPSimulator> flipSimulator = std::make_shared<FLIPSimulator>(size, domain);

    RunSimulation(picSimulator, numberOfIterations, timeIntervalInSeconds, "pic_out", size, domain);
    RunSimulation(apicSimulator, numberOfIterations, timeIntervalInSeconds, "apic_out", size, domain);
    RunSimulation(flipSimulator, numberOfIterations, timeIntervalInSeconds, "flip_out", size, domain);
}