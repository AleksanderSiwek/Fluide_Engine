import os
import sys

sys.path.append("D:/_STUDIA/Praca_magisterska/Fluid_Engine/build/src/python_wrapper\Debug/")
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import PyFluidEngine as pyf

timeIntervalInSeconds = 1.0 / 30.0

frame = pyf.Frame(timeIntervalInSeconds)
domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(4.0, 8.0, 8.0))
mesh = pyf.TriangleMesh()
gravity = pyf.DirectionalField(pyf.Vector3D(0.0, -9.8, 0.0))
objLoader = pyf.OBJManager()

objLoader.Load("../test_cases/short_water_wall.obj", mesh)

simulator = pyf.PICSimulator(pyf.Vector3S(20, 40, 40), domain);
simulator.AddExternalForce(gravity)
simulator.SetViscosity(0)
simulator.SetMaxClf(3)
simulator.SetMaxClf(3)
simulator.SetCurrentFrame(frame)
simulator.InitializeFromTriangleMesh(mesh)

for i in range(25):
    print("Iteration = ", i)
    simulator.AdvanceSingleFrame()
    simulator.GetSurface(mesh)
    objLoader.Save("../../" + "python_pic_test" + "_" + str(i) + ".obj", mesh)
    mesh.Clear();

print(simulator)
# print(pyf.PICSimulator(pyf.Vector3(1, 1, 1), domain))