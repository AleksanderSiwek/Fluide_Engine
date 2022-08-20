import os
import sys

sys.path.append("D:/_STUDIA/Praca_magisterska/Fluid_Engine/build/src/python_wrapper\Debug/")
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import PyFluidEngine as pyf

def RunTest_1():
    print("Runing test 1...\n")
    
    timeIntervalInSeconds = 1.0 / 45.0
    numberOfIterations = 360

    domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(5.0, 5.0, 10.0))
    size = pyf.Vector3S(60, 60, 120)

    picSimulator = pyf.PICSimulator(size, domain);
    apicSimulator = pyf.APICSimulator(size, domain);
    flipSimulator = pyf.FLIPSimulator(size, domain);

    #RunTestSimulation_1(apicSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/2_walls_2_apic", "apic", size, domain)
    # RunTestSimulation_1(flipSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/2_walls_2_flip", "flip", size, domain)
    RunTestSimulation_1(picSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/2_walls_2_pic", "pic", size, domain)

def RunTest_2():
    print("Runing test 2...\n")
    
    timeIntervalInSeconds = 1.0 / 60.0
    numberOfIterations = 480

    domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(4.0, 4.0, 8.0))
    size = pyf.Vector3S(60, 60, 120)

    picSimulator = pyf.PICSimulator(size, domain);
    apicSimulator = pyf.APICSimulator(size, domain);
    flipSimulator = pyf.FLIPSimulator(size, domain);

    # RunTestSimulation_2(apicSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/collider_apic", "apic", size, domain)
    RunTestSimulation_2(flipSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/collider_flip", "flip", size, domain)
    RunTestSimulation_2(picSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/collider_pic", "pic", size, domain)

def RunTest_3():
    print("Runing test 3...\n")
    
    timeIntervalInSeconds = 1.0 / 30.0
    numberOfIterations = 360

    domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(4.0, 4.0, 4.0))
    size = pyf.Vector3S(80, 80, 80)

    picSimulator = pyf.PICSimulator(size, domain);
    apicSimulator = pyf.APICSimulator(size, domain);
    flipSimulator = pyf.FLIPSimulator(size, domain);

    RunTestSimulation_3(apicSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/surface_field_apic", "apic", size, domain, size)
    RunTestSimulation_3(flipSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/surface_field_flip", "flip", size, domain, size)
    # RunTestSimulation_3(picSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/surface_field_pic", "pic", size, domain, size)



def RunTestSimulation_1(simulator, numberOfIterations, timeIntervalInSeconds, cacheDir, outFilePrefix, size, domain):
    # Load Initial state
    fluidMesh = pyf.TriangleMesh()
    objLoader = pyf.OBJManager()
    objLoader.Load("../test_cases/2_short_walls.obj", fluidMesh)

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


def RunTestSimulation_2(simulator, numberOfIterations, timeIntervalInSeconds, cacheDir, outFilePrefix, size, domain):
    # Load Initial state
    fluidMesh = pyf.TriangleMesh()
    objLoader = pyf.OBJManager()
    objLoader.Load("../test_cases/short_water_wall.obj", fluidMesh)
    
   # Setup colliders
    colliderMesh_1 = pyf.TriangleMesh()
    objLoader.Load("../test_cases/collider_2.obj", colliderMesh_1)
    collider_1 = pyf.TriangleMeshCollider(simulator.GetResolution(), simulator.GetOrigin(), simulator.GetGridSpacing(), colliderMesh_1)
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

def RunTestSimulation_3(simulator, numberOfIterations, timeIntervalInSeconds, cacheDir, outFilePrefix, size, domain, resolution):
    # Load Initial state
    fluidMesh = pyf.TriangleMesh()
    objLoader = pyf.OBJManager()
    objLoader.Load("../test_cases/small_water_pool.obj", fluidMesh)

    # Setup Simulator
    forceMesh = pyf.TriangleMesh()
    objLoader.Load("../test_cases/monkey.obj", forceMesh)
    #simulator.AddExternalForce(pyf.DirectionalField(pyf.Vector3D(0.0, -9.8, 0.0)))
    simulator.AddExternalForce(pyf.VolumeField(forceMesh, resolution, domain, -10))
    simulator.SetViscosity(0.01)
    simulator.SetMaxClf(5.0)
    simulator.InitializeFromTriangleMesh(fluidMesh)

    # Run simulation
    runner = pyf.SimulationRunner()
    runner.RunSimulation(simulator, timeIntervalInSeconds, numberOfIterations, outFilePrefix, cacheDir)
    print("Simulation ended!")
    runner.GetStats().PrintStats()
    print("\n")

# RunTest_1()
# RunTest_2()
RunTest_3()