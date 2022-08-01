#include "pic_simulator.hpp"

#include <random>
#include <concurrent_vector.h>  

#include "common/cuda_array_utils.hpp"
#include "fluid_solvers/cuda_blocked_boundry_condition_solver.hpp"

// TO DO: DELETE
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

PICSimulator::PICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
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
    _fluid.viscosity = 0.5;
    _fluid.density = 1;

    _maxClf = 1;
    _particlesPerBlok = 12;

    SetDiffusionSolver(std::make_shared<BackwardEulerDiffusionSolver>());
    SetPressureSolver(std::make_shared<SinglePhasePressureSolver>());
    _surfaceTracker = std::make_shared<MarchingCubesSolver>();
    _boundryConditionSolver = std::make_shared<CudaBlockedBoundryConditionSolver>();
    _boundryConditionSolver->SetColliders(std::make_shared<ColliderCollection>(gridSize, gridSpacing, domain.GetOrigin()));
}

PICSimulator::~PICSimulator()
{

}

FluidMarkers PICSimulator::GetMarkers() const
{
    FluidMarkers markers;
    const auto& size = _fluid.sdf.GetSize();
    markers.Resize(size);
    markers.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(_fluid.sdf(i, j, k) < 0)
        {
            markers(i, j, k) = FLUID_MARK;
        }
        else
        {
            markers(i, j, k) = AIR_MARK;
        }
    });
    return markers;
}

void PICSimulator::PrintGrid(const Array3<double>& input) const
{
    const auto& size = input.GetSize();
    std::cout << std::setprecision(2) << std::fixed;
    for(size_t j = size.y ; j > 0; j--)
    {
        for(size_t k = 0; k < size.z; k++)
        {
            for(size_t i = 0; i < size.x; i++)
            {
                std::cout << input(i, j - 1, k) << " ";
            }
            std::cout << "      ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void PICSimulator::PrintMarkers() 
{
    auto markers = GetMarkers();
    const auto& size = markers.GetSize();
    for(size_t j = size.y ; j > 0; j--)
    {
        for(size_t k = 0; k < size.z; k++)
        {
            for(size_t i = 0; i < size.x; i++)
            {
                std::cout << ((markers(i, j - 1, k) == FLUID_MARK) ? "F" : "A") << " ";
            }
            std::cout << "      ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void PICSimulator::InitializeFrom3dMesh(const TriangleMesh& mesh)
{
    Mesh2SDF converter;
    converter.Build(mesh, _fluid.sdf);

    InitializeParticles();
}

void PICSimulator::AddExternalForce(const std::shared_ptr<ExternalForce> newForce)
{
    _externalForces.push_back(newForce);
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

void PICSimulator::SetColliders(std::shared_ptr<ColliderCollection> colliders)
{
    _boundryConditionSolver->SetColliders(colliders);
}

void PICSimulator::AddCollider(std::shared_ptr<Collider> collider)
{
    _boundryConditionSolver->AddCollider(collider);
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

void PICSimulator::GetSurface(TriangleMesh* mesh)
{
    _surfaceTracker->BuildSurface(_fluid.sdf, mesh);
}

const ScalarGrid3D& PICSimulator::GetFluidSdf() const
{
    return _fluid.sdf;
}

double PICSimulator::Cfl(double timeIntervalInSceonds) const
{
    const auto& size = _fluid.velocityGrid.GetSize();
    const auto& gridSpacing = _fluid.velocityGrid.GetGridSpacing();
    const auto& velocityGrid = _fluid.velocityGrid;
    double maxVelocity = 0.0;
    velocityGrid.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> vect = _fluid.velocityGrid.ValueAtCellCenter(i, j, k).Abs() + timeIntervalInSceonds * Vector3<double>(0.0, -9.8, 0.0); 
        maxVelocity = std::max(maxVelocity, vect.x);
        maxVelocity = std::max(maxVelocity, vect.y);
        maxVelocity = std::max(maxVelocity, vect.z);
    });

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
    std::cout << "Number of particles: " << _fluid.particleSystem.GetParticleNumber() << "\n";

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
}

void PICSimulator::OnBeginAdvanceTimeStep(double timeIntervalInSeconds)
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

void PICSimulator::OnEndAdvanceTimeStep(double timeIntervalInSeconds)
{

}

void PICSimulator::ComputeExternalForces(double timeIntervalInSeconds)
{
    const auto& size = _fluid.velocityGrid.GetSize();
    auto& velGrid = _fluid.velocityGrid;
    for(size_t forceIdx = 0; forceIdx < _externalForces.size(); forceIdx++)
    {
        _externalForces[forceIdx]->ApplyExternalForce(velGrid, timeIntervalInSeconds);
    }
    ApplyBoundryCondition();
}

void PICSimulator::ComputeDiffusion(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _diffusionSolver->Solve(currentVelocity, _fluid.sdf, _boundryConditionSolver->GetColliderSdf(), timeIntervalInSeconds, _fluid.viscosity, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void PICSimulator::ComputePressure(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _pressureSolver->Solve(currentVelocity, _fluid.sdf, _boundryConditionSolver->GetColliderSdf(), timeIntervalInSeconds, _fluid.density, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void PICSimulator::ComputeAdvection(double timeIntervalInSeconds)
{
    ExtrapolateVelocityToAir();
    ApplyBoundryCondition();
    TransferGrid2Particles();
    MoveParticles(timeIntervalInSeconds);
}

void PICSimulator::MoveParticles(double timeIntervalInSeconds)
{
    const auto& velGrid = _fluid.velocityGrid;
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);
    
    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        Vector3<double> pt0 = particlesPos[i];
        Vector3<double> pt1 = pt0;
        Vector3<double> vel = particlesVel[i];

        // Adaptive time-stepping
        unsigned int numSubSteps = static_cast<unsigned int>(std::max(_maxClf, 1.0));
        double dt = timeIntervalInSeconds / numSubSteps;
        for (unsigned int t = 0; t < numSubSteps; ++t) 
        {
            Vector3<double> vel0 = velGrid.Sample(pt0);

            // Mid-point rule
            Vector3<double> midPt = pt0 + 0.5 * dt * vel0;
            Vector3<double> midVel = velGrid.Sample(midPt);
            pt1 = pt0 + dt * midVel;

            pt0 = pt1;
        }

        if (pt1.x <= _domain.GetOrigin().x) {
            pt1.x = _domain.GetOrigin().x;
            vel.x = 0.0;
        }
        if (pt1.x >= _domain.GetOrigin().x + _domain.GetSize().x) {
            pt1.x = _domain.GetOrigin().x + _domain.GetSize().x;
            vel.x = 0.0;
        }
        if (pt1.y <= _domain.GetOrigin().y) {
            pt1.y = _domain.GetOrigin().y;
            vel.y = 0.0;
        }
        if (pt1.y >= _domain.GetOrigin().y + _domain.GetSize().y) {
            pt1.y = _domain.GetOrigin().y + _domain.GetSize().y;
            vel.y = 0.0;
        }
        if (pt1.z <= _domain.GetOrigin().z) {
            pt1.z = _domain.GetOrigin().z;
            vel.z = 0.0;
        }
        if (pt1.z >= _domain.GetOrigin().z + _domain.GetSize().z) {
            pt1.z = _domain.GetOrigin().z + _domain.GetSize().z;
            vel.z = 0.0;
        }

        particlesPos[i] = pt1;
        particlesVel[i] = vel;
    });

    const auto& colliderSdf = _boundryConditionSolver->GetColliderSdf();
    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        if(colliderSdf.Sample(particlesPos[i]) < 0)
        {
            auto& colliders = _boundryConditionSolver->GetColliders();
            for(size_t i = 0; i < colliders.size(); i++)
            {
                colliders[i]->ResolveCollision(0.0, 0.0, &particlesPos[i], &particlesVel[i]);
            }
        }
    });
}

void PICSimulator::ApplyBoundryCondition()
{
    _boundryConditionSolver->ConstrainVelocity(_fluid.velocityGrid, static_cast<size_t>(std::ceil(_maxClf)));
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

void PICSimulator::TransferParticles2Grid()
{
    auto& flow = _fluid.velocityGrid;
    const auto& size = flow.GetSize();
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& positions = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& velocities = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    // Clear velocity to zero
    flow.ParallelFill(0, 0, 0);

    // Weighted-average velocity
    auto& u = flow.GetDataXRef();
    auto& v = flow.GetDataYRef();
    auto& w = flow.GetDataZRef();
    Array3<double> uWeight(u.GetSize());
    Array3<double> vWeight(v.GetSize());
    Array3<double> wWeight(w.GetSize());
    auto& xMarkers = _fluid.xMarkers;
    auto& yMarkers = _fluid.yMarkers;
    auto& zMarkers = _fluid.zMarkers;
    xMarkers.Resize(size);
    yMarkers.Resize(size);
    zMarkers.Resize(size);
    xMarkers.ParallelFill(0);
    yMarkers.ParallelFill(0);
    zMarkers.ParallelFill(0);

    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        std::array<Vector3<size_t>, 8> indices;
        std::array<double, 8> weights;

        GetCooridnatesAndWeights(size, flow.GetDataXOrigin(), flow.GetGridSpacing(), positions[i], indices, weights);
        for (int j = 0; j < 8; ++j) 
        {
            u(indices[j]) += velocities[i].x * weights[j];
            uWeight(indices[j]) += weights[j];
            xMarkers(indices[j]) = 1;
        }

        GetCooridnatesAndWeights(size, flow.GetDataYOrigin(), flow.GetGridSpacing(), positions[i], indices, weights);
        for (int j = 0; j < 8; ++j) 
        {
            v(indices[j]) += velocities[i].y * weights[j];
            vWeight(indices[j]) += weights[j];
            yMarkers(indices[j]) = 1;
        }

        GetCooridnatesAndWeights(size, flow.GetDataZOrigin(), flow.GetGridSpacing(), positions[i], indices, weights);
        for (int j = 0; j < 8; ++j) 
        {
            w(indices[j]) += velocities[i].z * weights[j];
            wWeight(indices[j]) += weights[j];
            zMarkers(indices[j]) = 1;
        }
    });

    u.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if (uWeight(i, j, k) > 0.0) 
        {
            u(i, j, k) /= uWeight(i, j, k);
        }
    });
    
    v.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if (vWeight(i, j, k) > 0.0) 
        {
            v(i, j, k) /= vWeight(i, j, k);
        }
    });

    w.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if (wWeight(i, j, k) > 0.0) 
        {
            w(i, j, k) /= wWeight(i, j, k);
        }
    });
}

void PICSimulator::TransferGrid2Particles()
{
    auto& velGrid = _fluid.velocityGrid;
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        particlesVel[i] = velGrid.Sample(particlesPos[i]);
    });
}

void PICSimulator::BuildSignedDistanceField()
{
    auto& sdf = _fluid.sdf;
    const auto& sdfSize = sdf.GetSize();
    auto& particleSystem = _fluid.particleSystem;
    const auto& gridSpacing = sdf.GetGridSpacing();
    double maxH = std::max(gridSpacing.x, std::max(gridSpacing.y, gridSpacing.z));
    double radious = 1.2 * maxH / std::sqrt(2.0);
    double sdfBandRadious = 2.0 * radious;

    particleSystem.BuildSearcher(PARTICLE_POSITION_KEY, 2 * radious); 
    auto searcher = particleSystem.GetSearcher();
    sdf.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> gridPosition = sdf.GridIndexToPosition(i, j, k);
        double minDistance = sdfBandRadious;
        searcher->ForEachNearbyPoint(gridPosition, sdfBandRadious, [&](size_t, const Vector3<double>& x)
        {
            minDistance = std::min(minDistance, Collisions::DistanceToPoint(gridPosition, x));
        });
        sdf(i, j, k) = minDistance - radious;
    });
    //ExtrapolateIntoCollider();
}

void PICSimulator::ExtrapolateVelocityToAir()
{
    auto& markers = _fluid.markers;
    const auto& sdf = _fluid.sdf;
    const auto& size = markers.GetSize();
    auto& xVel = _fluid.velocityGrid.GetDataXRef();
    auto& yVel = _fluid.velocityGrid.GetDataYRef();
    auto& zVel = _fluid.velocityGrid.GetDataZRef();
    const auto prevX(xVel);
    const auto prevY(yVel);
    const auto prevZ(zVel);
    const auto& xMarker =_fluid.xMarkers;
    const auto& yMarker =_fluid.yMarkers;
    const auto& zMarker =_fluid.zMarkers;

    size_t numberOfIterations = static_cast<size_t>(std::ceil(MaxCfl()));
    WrappedCuda_ExtrapolateToRegion(prevX, xMarker, numberOfIterations, xVel);
    WrappedCuda_ExtrapolateToRegion(prevY, yMarker, numberOfIterations, yVel);
    WrappedCuda_ExtrapolateToRegion(prevZ, zMarker, numberOfIterations, zVel);
}

void PICSimulator::ExtrapolateIntoCollider()
{
    auto& sdf = _fluid.sdf;
    const auto& size = sdf.GetSize();

    size_t depth = static_cast<size_t>(std::ceil(_maxClf));
    const auto prevSdf(sdf);
    Array3<int> valid(size, 0);
    const auto& colliderSdf = _boundryConditionSolver->GetColliderSdf();
    valid.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> position = sdf.GridIndexToPosition(i, j, k);
        if(colliderSdf.Sample(position) < 0)
        {
            valid(i, j, k) = 0;
        }
        else
        {
            valid(i, j, k) = 1;
        }
    });
    WrappedCuda_ExtrapolateToRegion(prevSdf, valid, depth, sdf);
}

void PICSimulator::InitializeParticles()
{
    const auto& size = _fluid.sdf.GetSize();
    const auto& sdf = _fluid.sdf;
    const auto& velGrid = _fluid.velocityGrid;
    const auto& gridSpacing = velGrid.GetGridSpacing();
    auto& particles = _fluid.particleSystem;

    concurrency::concurrent_vector<Vector3<double>> positions;

    velGrid.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        const Vector3<double> pos = velGrid.GridIndexToPosition(i, j, k);
        if(sdf.Sample(pos) < 0)
        {
            // Initialzie particles
            for(size_t particleIdx = 0; particleIdx < _particlesPerBlok; particleIdx++)
            {
                double x = gridSpacing.x * ( (double)std::rand() / (double)RAND_MAX ) + pos.x;
                double y = gridSpacing.y * ( (double)std::rand() / (double)RAND_MAX ) + pos.y;
                double z = gridSpacing.z * ( (double)std::rand() / (double)RAND_MAX ) + pos.z;
                positions.push_back(Vector3<double>(x, y, z));
            }
        }
    });

    particles.AddParticles(positions.size(), std::vector<Vector3<double>>(positions.begin(), positions.end()), PARTICLE_POSITION_KEY);
}