#include "physics_animation.hpp"

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

void PhysicsAnimation::SetNumberOfSubTimesteps(size_t numberOfSubTimesteps)
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

size_t PhysicsAnimation::GetNumberOfSubTimeSteps() const
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

void PhysicsAnimation::AdvanceTimeStep(double timeIntervalInSeconds)
{
    _currentTime = _currentFrame.GetTimeInSeconds();

    const double subTimestepInterval = _currentFrame.GetTimeIntervalInSeconds() / _numberOfSubTimesteps;
    for(size_t i = 0; i < _numberOfSubTimesteps; i++)
    {
        OnAdvanceTimeStep(subTimestepInterval);
        _currentTime += subTimestepInterval;
    }
}

void PhysicsAnimation::Initialize()
{
    OnInitialize();
}

void PhysicsAnimation::OnInitialize()
{

}

