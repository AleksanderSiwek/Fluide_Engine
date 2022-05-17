#include <gtest/gtest.h>
#include "../src/particle_systems/hash_grid_particle_searcher.hpp"


TEST(HashGridParticleSearcher, HasNearbyPoint_Test)
{
    HashGridParticleSearcher searcher(Vector3<size_t>(7, 7, 7), 2);
    std::vector<Vector3<double>> particles;
    particles.push_back(Vector3<double>(-0.1, -0.1, -0.1));
    particles.push_back(Vector3<double>(-0.1, 1, -0.5));
    particles.push_back(Vector3<double>(0.5, 0.7, 0.5));
    particles.push_back(Vector3<double>(1, 1, 1));
    particles.push_back(Vector3<double>(1.1, 1.1, 1.1));
    particles.push_back(Vector3<double>(1.5, 0.5, 1));
    particles.push_back(Vector3<double>(2, 1, 1));
    particles.push_back(Vector3<double>(2, 2, 1));
    particles.push_back(Vector3<double>(2, 2, 2));
    particles.push_back(Vector3<double>(5, 5, 5));
    particles.push_back(Vector3<double>(5, 5, 5.001));
    particles.push_back(Vector3<double>(3.1, 2.1, 5));
    
    searcher.Build(particles);

    EXPECT_EQ(false, searcher.HasNearbyPoint(Vector3<double>(3.5, 3.5, 3.5), 1));
    EXPECT_EQ(true, searcher.HasNearbyPoint(Vector3<double>(5.5, 5.5, 5.5), 1));
    EXPECT_EQ(true, searcher.HasNearbyPoint(Vector3<double>(1, 1, 1), 1));
    EXPECT_EQ(true, searcher.HasNearbyPoint(Vector3<double>(1, 2, 3), 2));
}

TEST(HashGridParticleSearcher, GetNearbyPointsIndexes_Test)
{
    HashGridParticleSearcher searcher(Vector3<size_t>(6, 6, 6), 1);
    std::vector<Vector3<double>> particles;
    particles.push_back(Vector3<double>(0.5, 0.7, 0.5));
    particles.push_back(Vector3<double>(1, 1, 1));
    particles.push_back(Vector3<double>(1.1, 1.1, 1.1));
    particles.push_back(Vector3<double>(1.5, 0.5, 1));
    particles.push_back(Vector3<double>(2, 1, 1));
    particles.push_back(Vector3<double>(2, 2, 1));
    particles.push_back(Vector3<double>(2, 2, 2));
    particles.push_back(Vector3<double>(5, 5, 5));
    particles.push_back(Vector3<double>(5, 5, 5.001));
    particles.push_back(Vector3<double>(3.1, 2.1, 5));

    searcher.Build(particles);
    std::vector<size_t> nearbyIdx = searcher.GetNearbyPointsIndexes(Vector3<double>(5.5, 5.5, 5.5), 4.21);
    std::cout << "nearbyIdx size: " << nearbyIdx.size() << "\n";
    for(size_t i = 0; i < nearbyIdx.size(); i++)
    {
        std::cout << "point: (" << particles[nearbyIdx[i]].x << ", " << particles[nearbyIdx[i]].y << ", " << particles[nearbyIdx[i]].z << "), idx: " << nearbyIdx[i] << "\n";
    }
}

TEST(HashGridParticleSearcher, ForEachNearbyIdx_Test)
{
    std::vector<Vector3<double>> points;
    points.push_back(Vector3<double>(0, 1, 3));
    points.push_back(Vector3<double>(2, 5, 4));
    points.push_back(Vector3<double>(-1, 3, 0));

    std::cout << "Test" << "\n";
    HashGridParticleSearcher searcher(Vector3<size_t>(4, 4, 4), 2.0 * std::sqrt(10));
    searcher.Build(points);

    int cnt = 0;
    searcher.ForEachNearbyPoint(Vector3<double>(0, 0, 0), std::sqrt(10.0), [&](size_t i, const Vector3<double>& pt) 
    {
            EXPECT_TRUE(i == 0 || i == 2);
            if (i == 0) {
                EXPECT_EQ(points[0], pt);
            } else if (i == 2) {
                EXPECT_EQ(points[2], pt);
            }
            ++cnt;
        });
    EXPECT_EQ(2, cnt);   
}