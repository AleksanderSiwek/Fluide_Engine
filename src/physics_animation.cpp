#include "physics_animation.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>

PhysicsAnimation::PhysicsAnimation()
{
    Initialize();
}

PhysicsAnimation::~PhysicsAnimation()
{

}

void PhysicsAnimation::AdvanceSingleFrame()
{
    Frame f = _currentFrame;
    Update(++f);
}

void PhysicsAnimation::SetCurrentFrame(const Frame& frame)
{
    _currentFrame = frame;
}

void PhysicsAnimation::SetNumberOfSubTimesteps(unsigned int numberOfSubTimesteps)
{
    _numberOfSubTimesteps = numberOfSubTimesteps;
}

Frame PhysicsAnimation::GetCurrentFrame() const
{
    return _currentFrame;
}

double PhysicsAnimation::GetCurrentTimeInSeconds() const
{
    return _currentTime;
}

unsigned int PhysicsAnimation::GetNumberOfSubTimeSteps() const
{
    return _numberOfTimeSteps;
}

void PhysicsAnimation::OnUpdate(const Frame& frame)
{
    if(frame.GetIndex() > _currentFrame.GetIndex())
    {
        long int numberOfFrames = frame.GetIndex() - _currentFrame.GetIndex();
        
        for (size_t i = 0; i < numberOfFrames; ++i)
        {
            AdvanceTimeStep(frame.GetTimeIntervalInSeconds());
        }

        _currentFrame = frame;
    }
}

#define EPSYLON 0.0000000001
void PhysicsAnimation::AdvanceTimeStep(double timeIntervalInSeconds)
{
    auto globalStart = std::chrono::steady_clock::now();
    std::cout << std::setprecision(5) << std::fixed;
    std::cout << "========== ITERATION ==========\n";

    _currentTime = _currentFrame.GetTimeInSeconds();

    OnBeginIteration(timeIntervalInSeconds);

    double timeRemaining = timeIntervalInSeconds;
    size_t i = 0;
    while(timeRemaining > EPSYLON)
    {
        unsigned int numberOfSubTimesteps = NumberOfSubTimeSteps(timeRemaining);
        double subTimestepInterval = timeRemaining / static_cast<double>(numberOfSubTimesteps);
        std::cout << "SubStep number: " << i << "\n";
        OnAdvanceTimeStep(subTimestepInterval);
        timeRemaining -= subTimestepInterval;
        _currentTime += subTimestepInterval;
        i++;
        std::cout << "\n";
    }

    auto globalEnd = std::chrono::steady_clock::now();
    std::cout << "Iteration ended in: ";
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(globalEnd - globalStart).count() / 1000000000.0 << " [s]\n";
    std::cout << "================================\n\n";
}

void PhysicsAnimation::Initialize()
{
    OnInitialize();
}

void PhysicsAnimation::OnInitialize()
{

}

