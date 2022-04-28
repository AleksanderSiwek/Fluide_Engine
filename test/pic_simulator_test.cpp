#include <gtest/gtest.h>
#include "../src/pic_simulator.hpp"
#include "../src/3d/obj_manager.hpp"


void __PrintArray3(const Array3<double>& input)
{
    const auto& size = input.GetSize();
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Size: (" << size.x << ", " << size.y << ", " << size.z << ")\n";
    for(size_t j = size.y ; j > 0; j--)
    {
        for(size_t k = 0; k < size.z; k++)
        {
            for(size_t i = 0; i < size.x; i++)
            {
                std::cout << input(i, j - 1, k) << " ";
            }
            std::cout << "      ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void PrintMarkers(const FluidMarkers& input)
{
    const auto& size = input.GetSize();
    std::cout << std::setprecision(2) << std::fixed;
    for(size_t j = size.y ; j > 0; j--)
    {
        for(size_t k = 0; k < size.z; k++)
        {
            for(size_t i = 0; i < size.x; i++)
            {
                std::cout << ((input(i, j - 1, k) == FLUID_MARK) ? "F" : "A") << " ";
            }
            std::cout << "      ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

TEST(PICSimulatorTest, ExtrapolateToRegion_test)
{
    const Vector3<size_t> size(3, 3, 3);
    BoundingBox3D domain(Vector3<double>(0, 0, 0), Vector3<double>(1, 1, 1));
    PICSimulator simulator(size, domain);
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

TEST(PICSimulatorTest, Simulate_test)
{
    const Vector3<size_t> size(5, 5, 5);
    BoundingBox3D domain(Vector3<double>(0, 0, 0), Vector3<double>(5, 5, 5));

    TriangleMesh mesh;
    OBJManager objLoader;
    objLoader.Load("../../../test/test_cases/test_cube_222.obj", &mesh);

    PICSimulator simulator(size, domain);
    simulator.InitializeFrom3dMesh(mesh);

    Frame frame(0.1);
    simulator.SetCurrentFrame(frame);
    for(size_t i = 0; i < 15; i++)
    {
        std::cout << "Iteration = " << i << "\n";
        const auto& markers = simulator.GetMarkers();
        PrintMarkers(markers);
        simulator.AdvanceSingleFrame();
        std::cout << "\n\n";
    }
}