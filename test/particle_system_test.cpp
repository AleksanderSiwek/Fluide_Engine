#include <gtest/gtest.h>
#include "../src/particle_systems/particle_system.hpp"


TEST(ParticleSystemTest, DefaultConstructor_test)
{
    ParticleSystem particle_system;
    EXPECT_EQ(0, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(0, particle_system.GetVectorDataMaxIdx());
}

TEST(ParticleSystemTest, Resize_test)
{
    ParticleSystem particle_system;
    particle_system.Resize(500);
    particle_system.AddVectorValue("position");
    EXPECT_EQ(500, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(1, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(500, particle_system.GetVectorValues("position").size());
    EXPECT_EQ(500, particle_system.GetParticleNumber());
}

TEST(ParticleSystemTest, AddParticles_test)
{
    ParticleSystem particle_system;
    particle_system.AddVectorValue("position");
    particle_system.Resize(100);
    particle_system.AddParticles(100, std::vector<Vector3<double>>(100), "position");
    EXPECT_EQ(200, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(1, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(200, particle_system.GetVectorValues("position").size());
    EXPECT_EQ(200, particle_system.GetParticleNumber());
}

TEST(ParticleSystemTest, AddScalarValues_test)
{
    ParticleSystem particle_system;
    particle_system.Resize(100);
    particle_system.AddVectorValue("position");
    particle_system.AddScalarValue("density", 1);
    EXPECT_EQ(100, particle_system.GetParticleNumber());
    EXPECT_EQ(1, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(1, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(100, particle_system.GetVectorValues("position").size());
    EXPECT_EQ(100, particle_system.GetScalarValues("density").size());
    EXPECT_EQ(1, particle_system.GetScalarValues("density")[0]);    
}

TEST(ParticleSystemTest, AddVectorValues_test)
{
    ParticleSystem particle_system;
    particle_system.Resize(100);
    particle_system.AddVectorValue("position");
    particle_system.AddVectorValue("velocity");
    Vector3<double> initialValue(1, 1, 1);
    particle_system.AddVectorValue("velocity", initialValue);
    EXPECT_EQ(100, particle_system.GetParticleNumber());
    EXPECT_EQ(0, particle_system.GetScalarDataMaxIdx());
    EXPECT_EQ(2, particle_system.GetVectorDataMaxIdx());
    EXPECT_EQ(100, particle_system.GetVectorValues("position").size());
    EXPECT_EQ(100, particle_system.GetVectorValues("velocity").size());
    EXPECT_EQ(true, particle_system.GetVectorValues("velocity")[0] == initialValue);    
}

TEST(ParticleSystemTest, TODO_test)
{
    EXPECT_EQ(1, 0);
}

// TO DO 
