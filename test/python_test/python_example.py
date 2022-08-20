import os
import sys

# Add Fluid Engine as python package
sys.path.append("path/to/PyFluidEngine/package/")

# Add CUDA .dll's
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import PyFluidEngine as pyf


timeIntervalInSeconds = 1.0 / 45.0
numberOfIterations = 360

# Create simulator object
domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(5.0, 5.0, 10.0))
size = pyf.Vector3S(60, 60, 120)
simulator = pyf.PICSimulator(size, domain);

# Initialize fluid from 3D mesh
fluidMesh = pyf.TriangleMesh()
objLoader = pyf.OBJManager()
objLoader.Load("../test_cases/water_wall.obj", fluidMesh)
simulator.InitializeFromTriangleMesh(fluidMesh)

# Setup Simulator
simulator.AddExternalForce(pyf.DirectionalField(pyf.Vector3D(0.0, -9.8, 0.0)))
simulator.SetViscosity(0.01)
simulator.SetMaxCfl(4.0)

# Run simulation
runner = pyf.SimulationRunner()
runner.RunSimulation(simulator, timeIntervalInSeconds, numberOfIterations, "sim_output_", "cacheDir")
runner.GetStats().PrintStats()



