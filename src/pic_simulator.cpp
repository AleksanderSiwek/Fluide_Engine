#include "pic_simulator.hpp"

#include <random>
#include <concurrent_vector.h>  

#include "common/cuda_array_utils.hpp"
#include "fluid_solvers/cuda_blocked_boundry_condition_solver.hpp"
#include "fluid_solvers/blocked_boundry_condition_solver.hpp"
#include "fluid_solvers/cuda_blocked_boundry_condition_solver.hpp"

// TO DO: DELETE
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

PICSimulator::PICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
    : HybridSimulator(gridSize, domain)
{
    SetDiffusionSolver(std::make_shared<BackwardEulerDiffusionSolver>());
    SetPressureSolver(std::make_shared<SinglePhasePressureSolver>());
    _surfaceTracker = std::make_shared<MarchingCubesSolver>();
    _boundryConditionSolver = std::make_shared<BlockedBoundryConditionSolver>();
    _boundryConditionSolver->SetColliders(std::make_shared<ColliderCollection>(gridSize, GetGridSpacing(), GetOrigin()));
    _maxParticleSpeedInCfl = 100;
}

PICSimulator::~PICSimulator()
{

}

void PICSimulator::InitializeFromTriangleMesh(const TriangleMesh& mesh)
{
    Mesh2SDF converter;
    converter.Build(mesh, _fluid.sdf);

    InitializeParticles();
}

void PICSimulator::AddExternalForce(const std::shared_ptr<ExternalForce> newForce)
{
    _externalForces.push_back(newForce);
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

void PICSimulator::GetSurface(TriangleMesh& mesh)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "Surface Tracker: ";
    _surfaceTracker->BuildSurface(_fluid.sdf, mesh);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";
}

double PICSimulator::Cfl(double timeIntervalInSeconds) const
{
    const auto& size = _fluid.velocityGrid.GetSize();
    const auto& gridSpacing = _fluid.velocityGrid.GetGridSpacing();
    const auto& velocityGrid = _fluid.velocityGrid;
    double maxVelocity = 0.0;
    velocityGrid.ForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> vect = _fluid.velocityGrid.ValueAtCellCenter(i, j, k).Abs() + timeIntervalInSeconds * Vector3<double>(0.0, -9.8, 0.0); 
        maxVelocity = std::max(maxVelocity, vect.x);
        maxVelocity = std::max(maxVelocity, vect.y);
        maxVelocity = std::max(maxVelocity, vect.z);
    });

    double minGridSize = std::min(gridSpacing.x, std::min(gridSpacing.y, gridSpacing.z));
    double cflVal = maxVelocity * timeIntervalInSeconds / minGridSize;
    return cflVal;
}

void PICSimulator::OnInitialize()
{

}

void PICSimulator::OnBeginIteration(double timeIntervalInSeconds)
{
    std::cout << "OnBeginIteration!\n";
    // DumpParticlesSpeed(timeIntervalInSeconds);
}


void PICSimulator::OnAdvanceTimeStep(double timeIntervalInSeconds)
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
        unsigned int numSubSteps = static_cast<unsigned int>(std::max(_maxCfl, 1.0));
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

void PICSimulator::DumpParticlesSpeed(double timeIntervalInSeconds)
{
    // size_t dumpedParticlesCnt = 0;
    // size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    // auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);
    // const auto& gridSpacing = _fluid.velocityGrid.GetGridSpacing();
    // double minGridSize = std::min(gridSpacing.x, std::min(gridSpacing.y, gridSpacing.z));
    // _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    // {
    //     double particleSpeedInCfl = ((particlesVel[i] + particlesVel[i] * timeIntervalInSeconds) / minGridSize).GetLength();
    //     if(particleSpeedInCfl > _maxParticleSpeedInCfl)
    //     {
    //         particlesVel[i] = particlesVel[i] / particleSpeedInCfl * _maxParticleSpeedInCfl;
    //         dumpedParticlesCnt++;
    //     }
    // });
    // std::cout << "\n" << "Dumped particles: " << dumpedParticlesCnt / (double)numberOfParticles * 100.0 << "%" << "\n";
}


void PICSimulator::ApplyBoundryCondition()
{
    _boundryConditionSolver->ConstrainVelocity(_fluid.velocityGrid, static_cast<size_t>(std::ceil(_maxCfl)));
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
    std::cout << "Current CFL: " << currentCfl << "\n";
    return static_cast<unsigned int>(std::max(std::ceil(currentCfl / _maxCfl), 1.0));
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
    xMarkers.Resize(u.GetSize());
    yMarkers.Resize(v.GetSize());
    zMarkers.Resize(w.GetSize());
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
    ExtrapolateToRegion(prevX, xMarker, numberOfIterations, xVel);
    ExtrapolateToRegion(prevY, yMarker, numberOfIterations, yVel);
    ExtrapolateToRegion(prevZ, zMarker, numberOfIterations, zVel);
}

void PICSimulator::ExtrapolateIntoCollider()
{
    auto& sdf = _fluid.sdf;
    const auto& size = sdf.GetSize();

    size_t depth = static_cast<size_t>(std::ceil(_maxCfl));
    const auto prevSdf(sdf);
    Array3<int> valid(size, 0);
    // const auto& colliderSdf = _boundryConditionSolver->GetColliderSdf();
    // valid.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    // {
    //     Vector3<double> position = sdf.GridIndexToPosition(i, j, k);
    //     if(colliderSdf.Sample(position) < 0)
    //     {
    //         valid(i, j, k) = 0;
    //     }
    //     else
    //     {
    //         valid(i, j, k) = 1;
    //     }
    // });
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