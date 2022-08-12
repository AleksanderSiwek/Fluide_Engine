import os
import sys

sys.path.append("D:/_STUDIA/Praca_magisterska/Fluid_Engine/build/src/python_wrapper\Debug/")
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import PyFluidEngine as pyf

def RunTestSimulation(simulator, numberOfIterations, timeIntervalInSeconds, cacheDir, outFilePrefix, size, domain):
    # Load Initial state
    fluidMesh = pyf.TriangleMesh()
    objLoader = pyf.OBJManager()
    objLoader.Load("../test_cases/water_wall.obj", fluidMesh)

    # Setup colliders
    colliderMesh_1 = pyf.TriangleMesh()
    colliderMesh_2 = pyf.TriangleMesh()
    objLoader.Load("../test_cases/collider_2.obj", colliderMesh_1)
    objLoader.Load("../test_cases/collider_2.obj", colliderMesh_2)
    collider_1 = pyf.TriangleMeshCollider(simulator.GetResolution(), simulator.GetOrigin(), simulator.GetGridSpacing(), colliderMesh_1)
    collider_2 = pyf.TriangleMeshCollider(simulator.GetResolution(), simulator.GetOrigin(), simulator.GetGridSpacing(), colliderMesh_2)
    simulator.AddCollider(collider_1)

    # Setup Simulator
    simulator.AddExternalForce(pyf.DirectionalField(pyf.Vector3D(0.0, -9.8, 0.0)))
    simulator.SetViscosity(0.01)
    simulator.SetMaxClf(4.0)
    simulator.InitializeFromTriangleMesh(fluidMesh)

    # Run simulation
    runner = pyf.SimulationRunner()
    runner.RunSimulation(simulator, timeIntervalInSeconds, numberOfIterations, outFilePrefix, cacheDir)
    print("Simulation ended!")
    runner.GetStats().PrintStats()
    print("\n")


timeIntervalInSeconds = 1.0 / 60.0
numberOfIterations = 360

domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(4.0, 4.5, 8.0))
size = pyf.Vector3S(60, 75, 120)

picSimulator = pyf.PICSimulator(size, domain);
apicSimulator = pyf.APICSimulator(size, domain);
flipSimulator = pyf.FLIPSimulator(size, domain);

RunTestSimulation(apicSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/collider_apic_py_test", "apic", size, domain)
RunTestSimulation(flipSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/collider_flip_py_test", "flip", size, domain)
#RunTestSimulation(picSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/collider_pic_py_test", "pic", size, domain)

