import os
import sys

sys.path.append("D:/_STUDIA/Praca_magisterska/Fluid_Engine/build/src/python_wrapper\Debug/")
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import PyFluidEngine as pyf

def RunTestSimulation(simulator, numberOfIterations, timeIntervalInSeconds, cacheDir, outFilePrefix, size, domain):
    # Load Initial state
    fluidMesh = pyf.TriangleMesh()
    objLoader = pyf.OBJManager()
    objLoader.Load("../test_cases/short_water_wall.obj", fluidMesh)

    # Setup colliders
    colliderMesh_1 = pyf.TriangleMesh()
    colliderMesh_2 = pyf.TriangleMesh()
    objLoader.Load("../test_cases/collider_2.obj", colliderMesh_1)
    objLoader.Load("../test_cases/test_cube.obj", colliderMesh_2)

    # Setup Simulator
    simulator.AddExternalForce(pyf.DirectionalField(pyf.Vector3D(0.0, -9.8, 0.0)))
    simulator.SetViscosity(0)
    simulator.SetMaxClf(3)
    simulator.InitializeFromTriangleMesh(fluidMesh)

    # Run simulation
    runner = pyf.SimulationRunner()
    runner.RunSimulation(simulator, timeIntervalInSeconds, numberOfIterations, outFilePrefix, cacheDir)
    print("Simulation ended!")
    runner.GetStats().PrintStats()
    print("\n")


timeIntervalInSeconds = 1.0 / 30.0
numberOfIterations = 5

frame = pyf.Frame(timeIntervalInSeconds)
domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(4.0, 4.0, 4.0))
mesh = pyf.TriangleMesh()
gravity = pyf.DirectionalField(pyf.Vector3D(0.0, -9.8, 0.0))
objLoader = pyf.OBJManager()

objLoader.Load("../test_cases/short_water_wall.obj", mesh)
size = pyf.Vector3S(20, 20, 20)

picSimulator = pyf.PICSimulator(size, domain);
apicSimulator = pyf.APICSimulator(size, domain);
flipSimulator = pyf.FLIPSimulator(size, domain);

RunTestSimulation(picSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/pic_py_test", "pic", size, domain)
RunTestSimulation(apicSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/apic_py_test", "apic", size, domain)
RunTestSimulation(flipSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/flip_py_test", "flip", size, domain)