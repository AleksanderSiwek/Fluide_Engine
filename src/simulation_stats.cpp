#include "simulation_stats.hpp"

#include <iostream>
#include <numeric>
#include <algorithm>
#include <limits>


SimulationStats::SimulationStats()
{
    Clear();
}

SimulationStats::~SimulationStats()
{

}

void SimulationStats::PrintStats()
{
    std::cout << "================ SIMULATION STATS ================\n";
    std::cout << "Simulator type: " << simulatorType << "\n";
    std::cout << "Number of Iterations: " << numberOfIterations << "\n";
    std::cout << "Time interval: " << timeIntervalInSeconds << " [s]\n";
    std::cout << "Median frame: " << CalculateMedian(iterationTimeInSeconds) << " [s]\n";
    std::cout << "Mean frame: " << CalculateMean(iterationTimeInSeconds) << " [s]\n";
    std::cout << "Simulation time: " << simulationTimeInSeconds << " [s]\n";
    std::cout << "==================================================\n";
}

void SimulationStats::Clear()
{
    simulatorType = "";
    numberOfIterations = 0;
    timeIntervalInSeconds = 0;
    simulationTimeInSeconds = 0;
    iterationTimeInSeconds.clear();
}

double SimulationStats::CalculateMedian(std::vector<double> vector)
{
    if (vector.size() < 1)
        return std::numeric_limits<double>::signaling_NaN();

    const auto alpha = vector.begin();
    const auto omega = vector.end();

    const auto i1 = alpha + (vector.size()-1) / 2;
    const auto i2 = alpha + vector.size() / 2;

    std::nth_element(alpha, i1, omega);
    std::nth_element(i1, i2, omega);

    return 0.5 * (*i1 + *i2);
}

double SimulationStats::CalculateMean(std::vector<double> vector)
{
    if(vector.size() > 0)
    {
        return std::accumulate(vector.begin(), vector.end(), decltype(vector)::value_type(0)) / vector.size();
    }
    else
    {
        return 0;
    }
}