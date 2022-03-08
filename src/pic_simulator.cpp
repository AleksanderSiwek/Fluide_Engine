#include "pic_simulator.hpp"

PICSimulator::PICSimulator()
{

}

PICSimulator::~PICSimulator()
{

}

void PICSimulator::SetViscosity(double viscosity)
{
    _viscosity = viscosity;
}

void PICSimulator::SetDensity(double density)
{
    _density = density;
}

void PICSimulator::SetDiffusionSolver(std::shared_ptr<DiffusionSolver> diffusionSolver)
{
    _diffusionSolver = diffusionSolver;
}

void PICSimulator::SetPressureSolver(std::shared_ptr<PressureSolver> pressureSolver)
{
    _pressureSolver = pressureSolver;
}

double PICSimulator::GetViscosity() const
{
    return _viscosity;
}

double PICSimulator::GetDensity() const
{
    return _density;
}

void PICSimulator::OnInitialize()
{

}

void PICSimulator::OnAdvanceTimeStep(double timeIntervalInSeconds)
{
    BeginAdvanceTimeStep(timeIntervalInSeconds);

    ComputeExternalForces(timeIntervalInSeconds);

    ComputeDiffusion(timeIntervalInSeconds);
    ComputePressure(timeIntervalInSeconds);
    ComputeAdvection(timeIntervalInSeconds);

    EndAdvanceTimeStep(timeIntervalInSeconds);
}

void PICSimulator::OnBeginAdvanceTimeStep(double timeIntervalInSeconds)
{

}

void PICSimulator::OnEndAdvanceTimeStep(double timeIntervalInSeconds)
{

}

void PICSimulator::ComputeExternalForces(double timeIntervalInSeconds)
{

}

void PICSimulator::ComputeDiffusion(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_velocityGrid);
    
    _diffusionSolver->Solve(currentVelocity, FluidMarkers(_velocityGrid.GetSize()), timeIntervalInSeconds, &_velocityGrid);
}

void PICSimulator::ComputePressure(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_velocityGrid);
    
    _pressureSolver->Solve(currentVelocity, FluidMarkers(_velocityGrid.GetSize()), timeIntervalInSeconds, &_velocityGrid);
}

void PICSimulator::ComputeAdvection(double timeIntervalInSeconds)
{
    // TO DO
}

void PICSimulator::ApplyBoundryCondition()
{
    // TO DO
}

void PICSimulator::BeginAdvanceTimeStep(double tmeIntervalInSecons)
{
    OnBeginAdvanceTimeStep(tmeIntervalInSecons);
}

void PICSimulator::EndAdvanceTimeStep(double tmeIntervalInSecons)
{
    OnEndAdvanceTimeStep(tmeIntervalInSecons);
}

