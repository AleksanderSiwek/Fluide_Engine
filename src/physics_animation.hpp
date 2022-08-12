#ifndef PHYSICS_ANIMATION_HPP
#define PHYSICS_ANIMATION_HPP

#include "animation.hpp"

class PhysicsAnimation : public Animation
{
    public:
        PhysicsAnimation();
        virtual ~PhysicsAnimation();

        void AdvanceSingleFrame();

        Frame GetCurrentFrame() const;
        double GetCurrentTimeInSeconds() const;
        unsigned int GetNumberOfSubTimeSteps() const;

        void SetCurrentFrame(const Frame& frame);
        void SetNumberOfSubTimesteps(unsigned int numberOfSubTimesteps);

    protected:
        virtual void OnAdvanceTimeStep(double timeIntervalInSeconds) = 0;
        virtual void OnBeginIteration(double timeIntervalInSeconds) = 0;
        virtual void OnInitialize();
        virtual unsigned int NumberOfSubTimeSteps(double tmeIntervalInSecons) const = 0;

    private:
        Frame _currentFrame;
        unsigned int _numberOfTimeSteps = 1;
        double _currentTime = 0.0;
        float _stepTime = 1/60; // in seconds
        unsigned int _numberOfSubTimesteps;

        void OnUpdate(const Frame& frame) final;
        void AdvanceTimeStep(double timeIntervalInSeconds);
        void Initialize();

};

#endif // PHYSICS_ANIMATION_HPP