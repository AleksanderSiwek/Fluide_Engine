#include "pic_simulator.hpp"
#include "cuda_pic_simulator.hpp"
#include "flip_simulator.hpp"
#include "apic_simulator.hpp"
#include "3d/obj_manager.hpp"
#include "forces/directional_field.hpp"
#include "forces/point_field.hpp"
#include "3d/collider_collection.hpp"
#include "3d/triangle_mesh_collider.hpp"
#include "simulation_runner.hpp"
#include "emmiter/volume_emmiter.hpp"

#include <stdlib.h>  
#include <iostream>

void RunSimulation(std::shared_ptr<PICSimulator> simulator, size_t numberOfIterations, double timeIntervalInSeconds, std::string prefix, Vector3<size_t> size, BoundingBox3D domain)
{
    // Load fluid mesh
    TriangleMesh fluidMesh;
    OBJManager objLoader;
    objLoader.Load("../../../test/test_cases/short_water_wall.obj", fluidMesh);

    // Setup Emitter
    TriangleMesh emitterObject;
    objLoader.Load("../../../test/test_cases/emitter_1.obj", emitterObject);

    simulator->AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    //simulator->InitializeFromTriangleMesh(fluidMesh);
    simulator->AddEmitter(std::make_shared<VolumeEmitter>(emitterObject, size, domain, 12, Vector3<double>(0.0, 6.0, 0.0), 0.0));
    simulator->SetViscosity(0.01);
    simulator->SetMaxClf(5);

    SimulationRunner runner;
    runner.RunSimulation(simulator, timeIntervalInSeconds, numberOfIterations, prefix, "../../../simOutputs/" + prefix + "/");
    std::cout << "Simulation ended!\n";
    runner.GetStats().PrintStats();
    std::cout << "\n\n";
}

int main()
{
    size_t resolution = 30;
    Vector3<double> scalers(1, 1, 1);
    size_t numberOfIterations = 400;
    double timeIntervalInSeconds = 1.0 / 60.0;

    const Vector3<size_t> size((size_t)(resolution*scalers.x), (size_t)(resolution*scalers.y), (size_t)(resolution*scalers.z));
    Vector3<double> domainOrigin(0, 0, 0);
    Vector3<double> domainSize(6*scalers.x, 6*scalers.y, 6*scalers.z);
    BoundingBox3D domain(domainOrigin, domainSize);

    std::shared_ptr<PICSimulator> picSimulator = std::make_shared<PICSimulator>(size, domain);
    std::shared_ptr<APICSimulator> apicSimulator = std::make_shared<APICSimulator>(size, domain);
    std::shared_ptr<FLIPSimulator> flipSimulator = std::make_shared<FLIPSimulator>(size, domain);

    // RunSimulation(picSimulator, numberOfIterations, timeIntervalInSeconds, "pic_out", size, domain);
    // RunSimulation(apicSimulator, numberOfIterations, timeIntervalInSeconds, "apic_out", size, domain);
    RunSimulation(flipSimulator, numberOfIterations, timeIntervalInSeconds, "flip_out", size, domain);
}