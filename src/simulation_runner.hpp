#ifndef _SIMULATION_RUNNER_HPP
#define _SIMULATION_RUNNER_HPP

#include "hybrid_simulator.hpp"
#include "simulation_stats.hpp"

#include <memory>
#include <string>


class SimulationRunner
{
    public:
        SimulationRunner();

        ~SimulationRunner();

        void RunSimulation(const std::shared_ptr<HybridSimulator> simulator, double timeIntervalInSeconds, size_t numberOfIterations, std::string simulationOutputName, std::string cacheFolderPath);

        SimulationStats GetStats() const;

    private:
        SimulationStats _stats;
};

#endif // _SIMULATION_RUNNER_HPP