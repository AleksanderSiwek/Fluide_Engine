#include <gtest/gtest.h>
#include "../src/pic_simulator.hpp"


TEST(PICSimulatorTest, ExtrapolateToRegion_test)
{
    Vector3<size_t> size(3, 3, 3);
    PICSimulator simulator(size, Vector3<double>(1, 1, 1), Vector3<double>(0, 0, 0));
    FluidMarkers markers(size, AIR_MARK);
    Array3<double> vel(size, 0);
    vel(0, 0, 0) = 2;
    vel(1, 1, 0) = 8;
    vel(2, 2, 2) = 4;

    std::cout << "Start: \n";
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                std::cout << vel(i, j, k) << " ";
            }
            std::cout << "  ";
        }
        std::cout << "\n";
    }
    
    markers(0, 0, 0) = FLUID_MARK;
    markers(1, 1, 0) = FLUID_MARK;
    markers(2, 2, 2) = FLUID_MARK;

    simulator.ExtrapolateToRegion(vel, markers, 30);

    std::cout << "\nEnd: \n";
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                std::cout << vel(i, j, k) << " ";
            }
            std::cout << "  ";
        }
        std::cout << "\n";
    }
}