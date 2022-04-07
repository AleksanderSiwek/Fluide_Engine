#include <gtest/gtest.h>
#include "../src/particle_systems/hash_grid_particle_searcher.hpp"


TEST(HashGridParticleSearcher, HasNearbyPoint_Test)
{
    HashGridParticleSearcher searcher(Vector3<size_t>(5, 5, 5), 1);
    std::vector<Vector3<double>> particles;
    particles.push_back(Vector3<double>(0.5, 0.7, 0.5));
    particles.push_back(Vector3<double>(1, 1, 1));
    particles.push_back(Vector3<double>(1.1, 1.1, 1.1));
    particles.push_back(Vector3<double>(1.5, 0.5, 1));
    particles.push_back(Vector3<double>(2, 1, 1));
    particles.push_back(Vector3<double>(2, 2, 1));
    particles.push_back(Vector3<double>(2, 2, 2));
    particles.push_back(Vector3<double>(5, 5, 5));

    searcher.build(particles);

    EXPECT_EQ(false, searcher.HasNearbyPoint(Vector3<double>(10, 10, 10), 1));
    EXPECT_EQ(false, searcher.HasNearbyPoint(Vector3<double>(3.5, 3.5, 3.5), 1));
    EXPECT_EQ(true, searcher.HasNearbyPoint(Vector3<double>(5.5, 5.5, 5.5), 2));
    EXPECT_EQ(true, searcher.HasNearbyPoint(Vector3<double>(1, 1, 1), 1));
    //EXPECT_EQ(true, searcher.HasNearbyPoint(Vector3<double>(-1, -2, -2), 4.5));
    //EXPECT_EQ(true, searcher.HasNearbyPoint(Vector3<double>(1, 2, 3), 5));
}

TEST(HashGridParticleSearcher, GetNearbyPointsIndexes_Test)
{
    HashGridParticleSearcher searcher(Vector3<size_t>(5, 5, 5), 1);
    std::vector<Vector3<double>> particles;
    particles.push_back(Vector3<double>(0.5, 0.7, 0.5));
    particles.push_back(Vector3<double>(1, 1, 1));
    particles.push_back(Vector3<double>(1.1, 1.1, 1.1));
    particles.push_back(Vector3<double>(1.5, 0.5, 1));
    particles.push_back(Vector3<double>(2, 1, 1));
    particles.push_back(Vector3<double>(2, 2, 1));
    particles.push_back(Vector3<double>(2, 2, 2));
    particles.push_back(Vector3<double>(5, 5, 5));

    searcher.build(particles);
    std::vector<size_t> nearbyIdx = searcher.GetNearbyPointsIndexes(Vector3<double>(1, 1, 1), 1);
    for(size_t i = 0; i < nearbyIdx.size(); i++)
    {
        std::cout << "point: = (" << particles[nearbyIdx[i]].x << ", " << particles[nearbyIdx[i]].y << ", " << particles[nearbyIdx[i]].z << ")\n";
    }
}