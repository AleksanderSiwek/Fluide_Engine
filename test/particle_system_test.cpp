#include <gtest/gtest.h>
#include "../src/particle_systems/particle_system.hpp"


TEST(ParticleSystemTest, DefaultConstructor_test)
{
    ParticleSystem particle_system;
    EXPECT_EQ(0, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(1, particle_system.GetVectorDataMaxIdx());
}

TEST(ParticleSystemTest, Resize_test)
{
    ParticleSystem particle_system;
    particle_system.Resize(500);
    EXPECT_EQ(500, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(1, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(500, particle_system.GetVectorValue("position").size());
}

TEST(ParticleSystemTest, AddParticles_test)
{
    ParticleSystem particle_system;
    particle_system.Resize(100);
    particle_system.AddPartices(100, std::vector<Vector3<double>>(100));
    EXPECT_EQ(200, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(1, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(200, particle_system.GetVectorValue("position").size());
}

TEST(ParticleSystemTest, AddScalarValue_test)
{
    ParticleSystem particle_system;
    particle_system.Resize(100);
    particle_system.AddScalarValue("density", 1);
    EXPECT_EQ(100, particle_system.GetParticleNumber());
    EXPECT_EQ(1, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(1, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(100, particle_system.GetVectorValue("position").size());
    EXPECT_EQ(100, particle_system.GetScalarValue("density").size());
    EXPECT_EQ(1, particle_system.GetScalarValue("density")[0]);    
}

TEST(ParticleSystemTest, AddVectorValue_test)
{
    ParticleSystem particle_system;
    particle_system.Resize(100);
    Vector3<double> initialValue(1, 1, 1);
    particle_system.AddVectorValue("velocity", initialValue);
    EXPECT_EQ(100, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(2, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(100, particle_system.GetVectorValue("position").size());
    EXPECT_EQ(100, particle_system.GetVectorValue("velocity").size());
    EXPECT_EQ(true, particle_system.GetVectorValue("velocity")[0] == initialValue);    
}

// TO DO 
