#include "pic_simulator.hpp"

PICSimulator::PICSimulator()
{

}

PICSimulator::~PICSimulator()
{

}

void PICSimulator::SetViscosity(double viscosity)
{
    fluid.viscosity = viscosity;
}

void PICSimulator::SetDensity(double density)
{
    fluid.density = density;
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
    return fluid.viscosity;
}

double PICSimulator::GetDensity() const
{
    return fluid.density;
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
    FaceCenteredGrid3D currentVelocity(fluid.velocityGrid);
    
    _diffusionSolver->Solve(currentVelocity, FluidMarkers(fluid.velocityGrid.GetSize()), timeIntervalInSeconds, fluid.viscosity, &(fluid.velocityGrid));
}

void PICSimulator::ComputePressure(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(fluid.velocityGrid);
    
    _pressureSolver->Solve(currentVelocity, FluidMarkers(fluid.velocityGrid.GetSize()), timeIntervalInSeconds, fluid.density, &(fluid.velocityGrid));
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

