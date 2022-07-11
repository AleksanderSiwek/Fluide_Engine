#include <gtest/gtest.h>
#include "../src/pic_simulator.hpp"
#include "../src/3d/obj_manager.hpp"
#include "../src/forces/directional_field.hpp"
#include "../src/forces/point_field.hpp"

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

void PrintMinMax(const ScalarGrid3D& sdf)
{
    const auto& size = sdf.GetSize();
    double min = sdf(0, 0, 0);
    double max = sdf(0, 0, 0);

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                if(sdf(i, j, k) < min)
                {
                    min = sdf(i, j, k);
                }
                if(sdf(i, j, k) > max)
                {
                    max = sdf(i, j, k);
                }
            } 
        }
    }

    std::cout << "Min: " << min << "\n";
    std::cout << "Max: " << max << "\n";
}

TEST(PICSimulatorTest, ExtrapolateToRegion_test)
{
    const Vector3<size_t> size(3, 3, 3);
    BoundingBox3D domain(Vector3<double>(0, 0, 0), Vector3<double>(1, 1, 1));
    PICSimulator simulator(size, domain);
    Array3<int> markers(size, AIR_MARK);
    Array3<double> vel(size, 0);
    vel(0, 0, 0) = 2;
    vel(1, 1, 0) = 8;
    vel(1, 0, 0) = 4;

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
    
    markers(0, 0, 0) = 0;
    markers(1, 1, 0) = 0;
    markers(1, 0, 0) = 0;

    const auto prevVel(vel);
    ExtrapolateToRegion(prevVel, markers, 30, vel);

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
    const Vector3<size_t> size(80, 80, 80);
    BoundingBox3D domain(Vector3<double>(0, 0, 0), Vector3<double>(4, 4, 4));

    TriangleMesh mesh;
    OBJManager objLoader;
    objLoader.Load("../../../test/test_cases/water_wall.obj", &mesh);

    PICSimulator simulator(size, domain);
    simulator.AddExternalForce(std::make_shared<DirectionalField>(Vector3<double>(0, -9.81, 0)));
    simulator.AddExternalForce(std::make_shared<PointField>(Vector3<double>(2, 2, 2), 10));
    simulator.InitializeFrom3dMesh(mesh);
    simulator.SetViscosity(0.0);

    Frame frame(0.05);
    simulator.SetCurrentFrame(frame);
    for(size_t i = 0; i < 150; i++)
    {
        std::cout << "Iteration = " << i << "\n";
        simulator.AdvanceSingleFrame();
        simulator.GetSurface(&mesh);
        objLoader.Save("../../simulation_test_" + std::to_string(i) + ".obj", mesh);
        mesh.Clear();
        std::cout << "\n\n";
    }
}
