#ifndef _SIMULATION_STATS_HPP
#define _SIMULATION_STATS_HPP

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
        double timeIntervalInSeconds;
        double simulationTimeInSeconds;
        std::vector<double> iterationTimeInSeconds;   

    private:
        double CalculateMedian(std::vector<double> vector);
        double CalculateMean(std::vector<double> vector);
};

#endif // _SIMULATION_STATS_HPP