#include "hybrid_simulator.hpp"


HybridSimulator::HybridSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
    : _domain(domain)
{
    auto domainSize = _domain.GetSize();
    Vector3<double> gridSpacing(domainSize.x / gridSize.x, domainSize.y / gridSize.y, domainSize.z / gridSize.z);

    _fluid.velocityGrid.Resize(gridSize);
    _fluid.velocityGrid.SetGridSpacing(gridSpacing);
    _fluid.velocityGrid.SetOrigin(domain.GetOrigin());
    _fluid.sdf.Resize(gridSize);
    _fluid.sdf.SetGridSpacing(gridSpacing);
    _fluid.sdf.SetOrigin(domain.GetOrigin() + gridSpacing / 2.0);
    _fluid.markers.Resize(gridSize);
    _fluid.particleSystem.AddVectorValue(PARTICLE_POSITION_KEY);
    _fluid.particleSystem.AddVectorValue(PARTICLE_VELOCITY_KEY);

    _maxNumberOfSubSteps = 250;
    _maxCfl = 3;
    _particlesPerBlok = 8;
    _fluid.viscosity = 0;
    _fluid.density = 1;
}

HybridSimulator::~HybridSimulator()
{

}

void HybridSimulator::OnAdvanceTimeStep(double timeIntervalInSeconds) 
{

}

void HybridSimulator::OnInitialize()
{

}

unsigned int HybridSimulator::NumberOfSubTimeSteps(double tmeIntervalInSecons) const
{
    return 0;
}

double HybridSimulator::MaxCfl() const
{
    return _maxCfl;
}

double HybridSimulator::GetViscosity() const
{
    return _fluid.viscosity;
}

double HybridSimulator::GetDensity() const
{
    return _fluid.density;
}

size_t HybridSimulator::GetParticlesPerBlock() const
{
    return _particlesPerBlok;
}

Vector3<double> HybridSimulator::GetOrigin() const
{
    return _fluid.velocityGrid.GetOrigin();
}

Vector3<size_t> HybridSimulator::GetResolution() const
{
    return _fluid.velocityGrid.GetSize();
}   

Vector3<double> HybridSimulator::GetGridSpacing() const
{
    return _fluid.velocityGrid.GetGridSpacing();
}

const ScalarGrid3D& HybridSimulator::GetFluidSdf() const
{
    return _fluid.sdf;
}

void HybridSimulator::SetMaxClf(double maxCfl)
{
    _maxCfl = maxCfl;
}

void HybridSimulator::SetViscosity(double viscosity)
{
    _fluid.viscosity = viscosity;
}

void HybridSimulator::SetDensity(double density)
{
    _fluid.density = density;
}

void HybridSimulator::SetParticlesPerBlock(size_t particlesPerBlock)
{
    _particlesPerBlok = particlesPerBlock;
}
