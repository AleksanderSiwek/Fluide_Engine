#include "simulation_runner.hpp"

#include "3d/obj_manager.hpp"
#include <filesystem>


SimulationRunner::SimulationRunner()
{

}

SimulationRunner::~SimulationRunner()
{

}

void SimulationRunner::RunSimulation(std::shared_ptr<HybridSimulator> simulator, double timeIntervalInSeconds, size_t numberOfIterations, std::string simulationOutputName, std::string cacheFolderPath)
{
    OBJManager objManager;
    TriangleMesh fluidSurface;
    Frame simulatorFrame(timeIntervalInSeconds);
    simulator->SetCurrentFrame(simulatorFrame);

    std::filesystem::create_directories(cacheFolderPath);
    std::filesystem::create_directories(cacheFolderPath + "/simulated_frames");
    std::ofstream out(cacheFolderPath + "/" + "log.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();
    std::cout.rdbuf(out.rdbuf());

    for(size_t i = 0; i < numberOfIterations; i++)
    {
        auto start = std::chrono::steady_clock::now();
        simulator->AdvanceSingleFrame();
        simulator->GetSurface(fluidSurface);
        objManager.Save(cacheFolderPath + "/simulated_frames/" + simulationOutputName + "_" + std::to_string(i) + ".obj", fluidSurface);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0;
        _stats.simulationTimeInSeconds += duration;
        _stats.iterationTimeInSeconds.push_back(duration);
        fluidSurface.Clear();
        std::cout << std::endl;
    }

    _stats.numberOfIterations = numberOfIterations;
    _stats.timeIntervalInSeconds = timeIntervalInSeconds;
    _stats.simulatorType = typeid(*(simulator.get())).name();

    _stats.PrintStats();

    std::cout.rdbuf(coutbuf);
}

SimulationStats SimulationRunner::GetStats() const
{
    return _stats;
}
