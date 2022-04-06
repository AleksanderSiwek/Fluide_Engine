#include "pic_simulator.hpp"


PICSimulator::PICSimulator(const Vector3<size_t>& resolution, const Vector3<double> gridSpacing, const Vector3<double> gridOrigin)
{
    _fluid.velocityGrid.Resize(resolution);
    _fluid.velocityGrid.SetGridSpacing(gridSpacing);
    _fluid.velocityGrid.SetOrigin(gridOrigin);
    _fluid.particleSystem.AddVectorValue("position");
    _fluid.particleSystem.AddVectorValue("velocity");

    _maxClf = 5.0;

    SetDiffusionSolver(std::make_shared<BackwardEulerDiffusionSolver>());
    SetPressureSolver(std::make_shared<SinglePhasePressureSolver>());
}

PICSimulator::~PICSimulator()
{

}

void PICSimulator::SetViscosity(double viscosity)
{
    _fluid.viscosity = std::max(0.0, viscosity);
}

void PICSimulator::SetDensity(double density)
{
    _fluid.density = std::max(0.0, density);
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
    return _fluid.viscosity;
}

double PICSimulator::GetDensity() const
{
    return _fluid.density;
}

Vector3<double> PICSimulator::GetOrigin() const
{
    return _fluid.velocityGrid.GetOrigin();
}

Vector3<size_t> PICSimulator::GetResolution() const
{
    return _fluid.velocityGrid.GetSize();
}

Vector3<double> PICSimulator::GetGridSpacing() const
{
    return _fluid.velocityGrid.GetGridSpacing();
}

double PICSimulator::Cfl(double timeIntervalInSceonds) const
{
    const auto& size = _fluid.velocityGrid.GetSize();
    const auto& gridSpacing = _fluid.velocityGrid.GetGridSpacing();
    double maxVelocity = 0.0;
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
               Vector3<double> vect = _fluid.velocityGrid.ValueAtCellCenter(i, j, k) + timeIntervalInSceonds * Vector3<double>(0.0, -9.8, 0.0); 
               maxVelocity = std::max(maxVelocity, vect.x);
               maxVelocity = std::max(maxVelocity, vect.y);
               maxVelocity = std::max(maxVelocity, vect.z);
            }
        }
    }
    double minGridSize = std::min(gridSpacing.x, std::min(gridSpacing.y, gridSpacing.z));
    return maxVelocity * timeIntervalInSceonds / minGridSize;
}

double PICSimulator::MaxCfl() const
{
    return _maxClf;
}

void PICSimulator::SetMaxClf(double maxClf)
{
    _maxClf = maxClf;
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

    EndAdvanceTimeStep(timeIntervalInSeconds);
}

void PICSimulator::OnBeginAdvanceTimeStep(double timeIntervalInSeconds)
{

}

void PICSimulator::OnEndAdvanceTimeStep(double timeIntervalInSeconds)
{

}

// TO DO: do it for custom forces
void PICSimulator::ComputeExternalForces(double timeIntervalInSeconds)
{
    const auto& size = _fluid.velocityGrid.GetSize();
    auto& velX = _fluid.velocityGrid.GetDataXRef();
    auto& velY = _fluid.velocityGrid.GetDataXRef();
    auto& velZ = _fluid.velocityGrid.GetDataXRef();
    for(size_t forceIdx = 0; forceIdx < _externalForces.size(); forceIdx++)
    {
        for(size_t i = 0; i < size.x; i++)
        {
            for(size_t j = 0; j < size.y; j++)
            {
                for(size_t k = 0; k < size.z; k++)
                {
                    velX(i, j, k) += timeIntervalInSeconds * 0;
                    velY(i, j, k) += timeIntervalInSeconds * (-9.81);
                    velZ(i, j, k) += timeIntervalInSeconds * 0;
                    ApplyBoundryCondition();
                }
            }
        }
    }
}

void PICSimulator::ComputeDiffusion(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _diffusionSolver->Solve(currentVelocity, FluidMarkers(_fluid.velocityGrid.GetSize()), timeIntervalInSeconds, _fluid.viscosity, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void PICSimulator::ComputePressure(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _pressureSolver->Solve(currentVelocity, FluidMarkers(_fluid.velocityGrid.GetSize()), timeIntervalInSeconds, _fluid.density, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void PICSimulator::MoveParticles(double timeIntervalInSeconds)
{
    auto& velGrid = _fluid.velocityGrid;
    BoundingBox3D fluidBBox(velGrid.GetOrigin(), velGrid.GetDiemensions());
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(0);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(1);

    for(size_t i = 0; i < numberOfParticles; i++)
    {
        Vector3<double> pPos0 = particlesPos[i];
        Vector3<double> pPos1 = pPos0;
        Vector3<double> pVel1 = particlesVel[i];

        unsigned int numSubSteps = static_cast<unsigned int>(std::max(MaxCfl(), 1.0));
        double dt = timeIntervalInSeconds / numberOfParticles;
        for(unsigned int t = 0; t < numSubSteps; t++)
        {
            Vector3<double> pVel0 = velGrid.Sample(pPos0);
            Vector3<double> pMidPos = pPos0 + 0.5 * dt * pVel0;
            Vector3<double> pMidVel = velGrid.Sample(pMidPos);
            pPos1 = pPos0 + dt * pMidVel;
            pPos0 = pPos1;
        }

        if(pPos1.x <= fluidBBox.GetOrigin().x)
        {
            pPos1.x = fluidBBox.GetOrigin().x;
            pVel1.x = 0.0;
        }

        if(pPos1.x >= fluidBBox.GetOrigin().x + fluidBBox.GetSize().x)
        {
            pPos1.x = fluidBBox.GetOrigin().x + fluidBBox.GetSize().x;
            pVel1.x = 0.0;
        }

        if(pPos1.y <= fluidBBox.GetOrigin().y)
        {
            pPos1.y = fluidBBox.GetOrigin().y;
            pVel1.y = 0.0;
        }

        if(pPos1.y >= fluidBBox.GetOrigin().y + fluidBBox.GetSize().y)
        {
            pPos1.y = fluidBBox.GetOrigin().y + fluidBBox.GetSize().y;
            pVel1.y = 0.0;
        }

        if(pPos1.z <= fluidBBox.GetOrigin().z)
        {
            pPos1.z = fluidBBox.GetOrigin().z;
            pVel1.z = 0.0;
        }

        if(pPos1.z >= fluidBBox.GetOrigin().z + fluidBBox.GetSize().z)
        {
            pPos1.z = fluidBBox.GetOrigin().z + fluidBBox.GetSize().z;
            pVel1.z = 0.0;
        }
    
        particlesPos[i] = pPos1;
        particlesVel[i] = pVel1;
    }
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

unsigned int PICSimulator::NumberOfSubTimeSteps(double tmeIntervalInSecons) const
{
    double currentCfl = Cfl(tmeIntervalInSecons);
    return static_cast<unsigned int>(std::max(std::ceil(currentCfl / _maxClf), 1.0));
}

