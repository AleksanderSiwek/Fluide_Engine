#include "hybrid_simulator.hpp"


HybridSimulator::HybridSimulatorSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
    : _domain(domain)
{
    
}

HybridSimulator::~HybridSimulator()
{

}

void HybridSimulator::InitializeFrom3dMesh(const TriangleMesh& mesh)
{
    Mesh2SDF converter;
    converter.Build(mesh, _fluid.sdf);

    InitializeParticles();
}

void HybridSimulator::AddExternalForce(const std::shared_ptr<ExternalForce> newForce)
{
    _externalForces.push_back(newForce);
}

void HybridSimulator::SetViscosity(double viscosity)
{
    _fluid.viscosity = std::max(0.0, viscosity);
}

void HybridSimulator::SetDensity(double density)
{
    _fluid.density = std::max(0.0, density);
}

void HybridSimulator::SetDiffusionSolver(std::shared_ptr<DiffusionSolver> diffusionSolver)
{
    _diffusionSolver = diffusionSolver;
}

void HybridSimulator::SetPressureSolver(std::shared_ptr<PressureSolver> pressureSolver)
{
    _pressureSolver = pressureSolver;
}

void HybridSimulator::SetColliders(std::shared_ptr<ColliderCollection> colliders)
{
    _boundryConditionSolver->SetColliders(colliders);
}

void HybridSimulator::AddCollider(std::shared_ptr<Collider> collider)
{
    _boundryConditionSolver->AddCollider(collider);
}

double HybridSimulator::GetViscosity() const
{
    return _fluid.viscosity;
}

double HybridSimulator::GetDensity() const
{
    return _fluid.density;
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

void HybridSimulator::GetSurface(TriangleMesh* mesh)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "Surface Tracker: ";
    _surfaceTracker->BuildSurface(_fluid.sdf, mesh);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";
}

const ScalarGrid3D& HybridSimulator::GetFluidSdf() const
{
    return _fluid.sdf;
}

double HybridSimulator::MaxCfl() const
{
    return _maxClf;
}

void HybridSimulator::SetMaxClf(double maxClf)
{
    _maxClf = maxClf;
}

void HybridSimulator::OnInitialize()
{

}


void HybridSimulator::OnAdvanceTimeStep(double timeIntervalInSeconds)
{
    std::cout << "Number of particles: " << _fluid.particleSystem.GetParticleNumber() << "\n";
    auto startGlobal = std::chrono::steady_clock::now();

    auto start = std::chrono::steady_clock::now();
    std::cout << "BeginAdvanceTimeStep: ";
    BeginAdvanceTimeStep(timeIntervalInSeconds);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    start = std::chrono::steady_clock::now();
    std::cout << "External forces: ";
    ComputeExternalForces(timeIntervalInSeconds);
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    std::cout << "Diffusion: ";
    start = std::chrono::steady_clock::now();
    ComputeDiffusion(timeIntervalInSeconds);
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    std::cout << "Pressure: ";
    start = std::chrono::steady_clock::now();
    ComputePressure(timeIntervalInSeconds);
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    std::cout << "Advection: ";
    start = std::chrono::steady_clock::now();
    ComputeAdvection(timeIntervalInSeconds);
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    start = std::chrono::steady_clock::now();
    std::cout << "EndAdvanceTimeStep: ";
    EndAdvanceTimeStep(timeIntervalInSeconds);
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    end = std::chrono::steady_clock::now();
    std::cout << "SubStep ended in: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - startGlobal).count() / 1000000000.0 << " [s]\n";
}

void HybridSimulator::OnBeginAdvanceTimeStep(double timeIntervalInSeconds)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "BuildCollider(): ";
    _boundryConditionSolver->BuildCollider();
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    start = std::chrono::steady_clock::now();
    std::cout << "TransferParticles2Grid(): ";
    TransferParticles2Grid();
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    start = std::chrono::steady_clock::now();
    std::cout << "BuildSignedDistanceField(): ";
    BuildSignedDistanceField();
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    start = std::chrono::steady_clock::now();
    std::cout << "ExtrapolateVelocityToAir(): ";
    ExtrapolateVelocityToAir();
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    start = std::chrono::steady_clock::now();
    std::cout << "ApplyBoundryCondition(): ";
    ApplyBoundryCondition();
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";
}

void HybridSimulator::OnEndAdvanceTimeStep(double timeIntervalInSeconds)
{

}

void HybridSimulator::ComputeExternalForces(double timeIntervalInSeconds)
{
    const auto& size = _fluid.velocityGrid.GetSize();
    auto& velGrid = _fluid.velocityGrid;
    for(size_t forceIdx = 0; forceIdx < _externalForces.size(); forceIdx++)
    {
        _externalForces[forceIdx]->ApplyExternalForce(velGrid, timeIntervalInSeconds);
    }
    ApplyBoundryCondition();
}

void HybridSimulator::ComputeDiffusion(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _diffusionSolver->Solve(currentVelocity, _fluid.sdf, _boundryConditionSolver->GetColliderSdf(), timeIntervalInSeconds, _fluid.viscosity, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void HybridSimulator::ComputePressure(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _pressureSolver->Solve(currentVelocity, _fluid.sdf, _boundryConditionSolver->GetColliderSdf(), timeIntervalInSeconds, _fluid.density, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void HybridSimulator::ComputeAdvection(double timeIntervalInSeconds)
{
    ExtrapolateVelocityToAir();
    ApplyBoundryCondition();
    TransferGrid2Particles();
    MoveParticles(timeIntervalInSeconds);
}

void HybridSimulator::MoveParticles(double timeIntervalInSeconds)
{
    
}

void HybridSimulator::ApplyBoundryCondition()
{
    _boundryConditionSolver->ConstrainVelocity(_fluid.velocityGrid, static_cast<size_t>(std::ceil(_maxClf)));
}

void HybridSimulator::BeginAdvanceTimeStep(double tmeIntervalInSecons)
{
    OnBeginAdvanceTimeStep(tmeIntervalInSecons);
}

void HybridSimulator::EndAdvanceTimeStep(double tmeIntervalInSecons)
{
    OnEndAdvanceTimeStep(tmeIntervalInSecons);
}

unsigned int HybridSimulator::NumberOfSubTimeSteps(double tmeIntervalInSecons) const
{
    double currentCfl = Cfl(tmeIntervalInSecons);
    std::cout << "Current CFL: " << currentCfl << "\n";
    unsigned int numberOfSubSteps = static_cast<unsigned int>(std::max(std::ceil(currentCfl / _maxClf), 1.0));
    return std::min(numberOfSubSteps, _maxNumberOfSubSteps);
}
