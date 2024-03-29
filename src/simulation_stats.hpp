#ifndef _SIMULATION_STATS_HPP
#define _SIMULATION_STATS_HPP

#include "common/vector3.hpp"

#include <vector>
#include <string>


class SimulationStats
{
    public:
        SimulationStats();
        ~SimulationStats();

        void PrintStats();
        void Clear();

        std::string simulatorType;
        size_t numberOfIterations;
        Vector3<size_t> gridSize;
        size_t numberOfParticles;
        double timeIntervalInSeconds;
        double simulationTimeInSeconds;
        std::vector<double> iterationTimeInSeconds;   
        std::vector<double> cflPerIteration;

    private:
        double CalculateMedian(std::vector<double> vector);
        double CalculateMean(std::vector<double> vector);
};

#endif // _SIMULATION_STATS_HPP