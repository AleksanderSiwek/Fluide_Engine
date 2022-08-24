import os
import sys
import time


sys.path.append("D:/_STUDIA/Praca_magisterska/Fluid_Engine/build/src/python_wrapper\Debug/")
os.environ['PATH'] = os.path.join(os.environ['CUDA_PATH'], 'bin')

import PyFluidEngine as pyf
import pysdf

def RunTest_1():
    print("Runing test 1...\n")
    
    timeIntervalInSeconds = 1.0 / 60.0
    numberOfIterations = 300

    domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(6.0, 6.0, 6.0))
    size = pyf.Vector3S(100, 100, 100)

    picSimulator = pyf.PICSimulator(size, domain);
    apicSimulator = pyf.APICSimulator(size, domain);
    flipSimulator = pyf.FLIPSimulator(size, domain);

    RunTestSimulation_1(apicSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/volume_emitter_apic", "apic", size, domain)
    RunTestSimulation_1(flipSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/volume_emitter_flip", "flip", size, domain)
    RunTestSimulation_1(picSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/volume_emitter_pic", "pic", size, domain)

def RunTest_2():
    print("Runing test 2...\n")
    
    timeIntervalInSeconds = 1.0 / 60.0
    numberOfIterations = 600

    domain = pyf.BoundingBox3D(pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(3.0, 2.0, 1.0))
    size = pyf.Vector3S(150, 100, 50)

    picSimulator = pyf.PICSimulator(size, domain);
    apicSimulator = pyf.APICSimulator(size, domain);
    flipSimulator = pyf.FLIPSimulator(size, domain);
    RunTestSimulation_2(flipSimulator, numberOfIterations, timeIntervalInSeconds, "../../simOutputs/thriller_flip", "flip", size, domain)

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
    # Setup emitter
    emitterMesh = pyf.TriangleMesh()
    objLoader = pyf.OBJManager()
    objLoader.Load("../test_cases/emitter_1.obj", emitterMesh)
    simulator.AddEmitter(pyf.VolumeEmitter(emitterMesh, size, domain, 8, pyf.Vector3D(0.0, 6.0, 0.0), pyf.Vector3D(0.0, 0.0, 0.0)))


    # Setup Simulator
    simulator.AddExternalForce(pyf.DirectionalField(pyf.Vector3D(0.0, -9.8, 0.0)))
    simulator.SetViscosity(0.0)
    simulator.SetMaxClf(4.0)
    
    # Run simulation
    runner = pyf.SimulationRunner()
    runner.RunSimulation(simulator, timeIntervalInSeconds, numberOfIterations, outFilePrefix, cacheDir)
    print("Simulation ended!")
    runner.GetStats().PrintStats()
    print("\n")


def RunTestSimulation_2(simulator, numberOfIterations, timeIntervalInSeconds, cacheDir, outFilePrefix, size, domain):
    dance_animation_path = "D:/_STUDIA/Praca_magisterska/thriler_dance/"
    dance_animation = os.listdir(dance_animation_path)
    # Setup emitter
    emitterMesh = pyf.TriangleMesh()
    objLoader = pyf.OBJManager()
    objLoader.Load(dance_animation_path + dance_animation[0], emitterMesh)
    eimtter = pyf.VolumeEmitter(emitterMesh, size, domain, 8, pyf.Vector3D(0.0, 0.0, 0.0), pyf.Vector3D(0.0, 0.0, 0.0))
    simulator.AddEmitter(eimtter)
    
    # Setup Simulator
    simulator.AddExternalForce(pyf.DirectionalField(pyf.Vector3D(0.0, -14, 0.0)))
    simulator.SetViscosity(0.0)
    simulator.SetMaxClf(5.0)
    
    frame = pyf.Frame(timeIntervalInSeconds)
    outputMesh = pyf.TriangleMesh()
    simulator.SetCurrentFrame(frame)
    for i in range(numberOfIterations):
        start = time.perf_counter()
        print("Iteration: " + str(i))
        simulator.AdvanceSingleFrame()
        simulator.GetSurface(outputMesh)
        objLoader.Save(cacheDir + "/simulated_frames/" + outFilePrefix + "_" + str(i) + ".obj", outputMesh)
        outputMesh.Clear()
        if i % 2 == 0:
            emitterMesh.Clear()
            objLoader.Load(dance_animation_path + dance_animation[i], emitterMesh)
            eimtter.InitializeFromTriangleMesh(emitterMesh)
        end = time.perf_counter()
        print("Full iteration ended in: " + str(end - start) + "[s]\n")
    
    # Run simulation
    # runner = pyf.SimulationRunner()
    # runner.RunSimulation(simulator, timeIntervalInSeconds, numberOfIterations, outFilePrefix, cacheDir)
    # print("Simulation ended!")
    # runner.GetStats().PrintStats()

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

#RunTest_1()
RunTest_2()
# RunTest_3()