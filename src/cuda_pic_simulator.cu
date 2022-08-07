#include "cuda_pic_simulator.hpp"

#include <random>
#include <concurrent_vector.h>  

#include "common/cuda_array_utils.hpp"
#include "fluid_solvers/cuda_blocked_boundry_condition_solver.hpp"
#include "fluid_solvers/blocked_boundry_condition_solver.hpp"

// TO DO: DELETE
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>


// TO DO: Resarch tihs topic, this function is kind of slow
__device__ inline void atomicAddDouble(double *address, double value) 
{
    unsigned long long oldval, newval, readback;
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);

    while ((readback = atomicCAS((unsigned long long *) address, oldval, newval)) != oldval) 
    {
        oldval = readback;
        newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
}

__global__ void CUDA_TransferParticlesToGrid(double* xVelocity, double* yVelocity, double* zVelocity, 
                                            int* xMarkers, int* yMarkers, int* zMarkers, 
                                            double* xWeights, double* yWeights, double* zWeights, 
                                            CUDA_Vector3* particlePos, CUDA_Vector3* particleVel, CUDA_Int3 size, int numberOfParticles,
                                            CUDA_Vector3 xOrigin, CUDA_Vector3 yOrigin, CUDA_Vector3 zOrigin, 
                                            CUDA_Vector3 gridSpacing)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numberOfParticles)
    {
        xWeights[i] = 0;
        yWeights[i] = 0;
        zWeights[i] = 0;

        CUDA_Int3* indexes = (CUDA_Int3*)malloc(8 * sizeof(CUDA_Int3));
        double* weights = (double*)malloc(8 * sizeof(double));

        CUDA_GetCooridnatesAndWeights(size, xOrigin, gridSpacing, particlePos[i], indexes, weights);
        for (int j = 0; j < 8; ++j) 
        {
            int idx = indexes[i].x + size.x * (indexes[i].y + size.y * indexes[i].z);
            // atomicAddDouble(&(xVelocity[idx]), particleVel[i].x * weights[j]);
            // atomicAddDouble(&(xWeights[idx]), weights[j]);
            xVelocity[idx] += particleVel[i].x * weights[j];
            xWeights[idx] += weights[j];
            xMarkers[idx] = 1;
        }

        CUDA_GetCooridnatesAndWeights(size, yOrigin, gridSpacing, particlePos[i], indexes, weights);
        for (int j = 0; j < 8; ++j) 
        {
            int idx = indexes[i].x + size.x * (indexes[i].y + size.y * indexes[i].z);
            // atomicAddDouble(&(yVelocity[idx]), particleVel[i].y * weights[j]);
            // atomicAddDouble(&(yWeights[idx]), weights[j]);
            yVelocity[idx] += particleVel[i].y * weights[j];
            yWeights[idx] += weights[j];
            yMarkers[idx] = 1;
        }

        CUDA_GetCooridnatesAndWeights(size, zOrigin, gridSpacing, particlePos[i], indexes, weights);
        for (int j = 0; j < 8; ++j) 
        {
            int idx = indexes[i].x + size.x * (indexes[i].y + size.y * indexes[i].z);
            // atomicAddDouble(&(zVelocity[idx]), particleVel[i].z * weights[j]);
            // atomicAddDouble(&(zWeights[idx]), weights[j]);
            zVelocity[idx] += particleVel[i].z * weights[j];
            zWeights[idx] += weights[j];
            zMarkers[idx] = 1;
        }
    }
}

__global__ void CUDA_DivideGridByWeights(double* xVelocity, double* yVelocity, double* zVelocity, double* xWeights, double* yWeights, double* zWeights, CUDA_Int3 size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        if(xWeights[idx] > 0.0) 
        {
            xVelocity[idx] /= xWeights[idx];
        }

        if(yWeights[idx] > 0.0) 
        {
            yVelocity[idx] /= yWeights[idx];
        }

        if(zWeights[idx] > 0.0) 
        {
            zVelocity[idx] /= zWeights[idx];
        }
    }
}

CudaPICSimulator::CudaPICSimulator() : _wasCudaInitialized(false)
{

}

CudaPICSimulator::CudaPICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
    : _domain(domain), _wasCudaInitialized(false)
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
    _fluid.xMarkers.Resize(gridSize);
    _fluid.yMarkers.Resize(gridSize);
    _fluid.zMarkers.Resize(gridSize);
    _fluid.particleSystem.AddVectorValue(PARTICLE_POSITION_KEY);
    _fluid.particleSystem.AddVectorValue(PARTICLE_VELOCITY_KEY);
    _fluid.viscosity = 0.5;
    _fluid.density = 1;
    
    _maxClf = 1;
    _particlesPerBlok = 8;

    SetDiffusionSolver(std::make_shared<BackwardEulerDiffusionSolver>());
    SetPressureSolver(std::make_shared<SinglePhasePressureSolver>());
    _surfaceTracker = std::make_shared<MarchingCubesSolver>();
    _boundryConditionSolver = std::make_shared<CudaBlockedBoundryConditionSolver>();
    _boundryConditionSolver->SetColliders(std::make_shared<ColliderCollection>(gridSize, gridSpacing, domain.GetOrigin()));

    _maxNumberOfSubSteps = 100;
    _cflTolerance = 0.1;
}

CudaPICSimulator::~CudaPICSimulator()
{
    if(_wasCudaInitialized)
    {
        FreeCudaMemory();
    }
}

void CudaPICSimulator::InitializeFrom3dMesh(const TriangleMesh& mesh)
{
    Mesh2SDF converter;
    converter.Build(mesh, _fluid.sdf);
    InitializeParticles();
    ReAllocateCudaMemory();
    CopyFluidStateHostToDevice();
}

void CudaPICSimulator::AddExternalForce(const std::shared_ptr<ExternalForce> newForce)
{
    _externalForces.push_back(newForce);
}

void CudaPICSimulator::SetViscosity(double viscosity)
{
    _fluid.viscosity = std::max(0.0, viscosity);
}

void CudaPICSimulator::SetDensity(double density)
{
    _fluid.density = std::max(0.0, density);
}

void CudaPICSimulator::SetDiffusionSolver(std::shared_ptr<DiffusionSolver> diffusionSolver)
{
    _diffusionSolver = diffusionSolver;
}

void CudaPICSimulator::SetPressureSolver(std::shared_ptr<PressureSolver> pressureSolver)
{
    _pressureSolver = pressureSolver;
}

void CudaPICSimulator::SetColliders(std::shared_ptr<ColliderCollection> colliders)
{
    _boundryConditionSolver->SetColliders(colliders);
}

void CudaPICSimulator::AddCollider(std::shared_ptr<Collider> collider)
{
    _boundryConditionSolver->AddCollider(collider);
}

double CudaPICSimulator::GetViscosity() const
{
    return _fluid.viscosity;
}

double CudaPICSimulator::GetDensity() const
{
    return _fluid.density;
}

Vector3<double> CudaPICSimulator::GetOrigin() const
{
    return _fluid.velocityGrid.GetOrigin();
}

Vector3<size_t> CudaPICSimulator::GetResolution() const
{
    return _fluid.velocityGrid.GetSize();
}

Vector3<double> CudaPICSimulator::GetGridSpacing() const
{
    return _fluid.velocityGrid.GetGridSpacing();
}

void CudaPICSimulator::GetSurface(TriangleMesh* mesh)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "Surface Tracker: ";
    _surfaceTracker->BuildSurface(_fluid.sdf, mesh);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";
}

const ScalarGrid3D& CudaPICSimulator::GetFluidSdf() const
{
    return _fluid.sdf;
}

double CudaPICSimulator::Cfl(double timeIntervalInSeconds) const
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
    return cflVal + cflVal * _cflTolerance;
}

double CudaPICSimulator::MaxCfl() const
{
    return _maxClf;
}

void CudaPICSimulator::SetMaxClf(double maxClf)
{
    _maxClf = maxClf;
}

void CudaPICSimulator::OnInitialize()
{

}

void CudaPICSimulator::OnAdvanceTimeStep(double timeIntervalInSeconds)
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

void CudaPICSimulator::OnBeginAdvanceTimeStep(double timeIntervalInSeconds)
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

    CopyFluidStateHostToDevice();

    start = std::chrono::steady_clock::now();
    std::cout << "BuildSignedDistanceField(): ";
    BuildSignedDistanceField();
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    CopyFluidStateDeviceToHost();

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

void CudaPICSimulator::OnEndAdvanceTimeStep(double timeIntervalInSeconds)
{

}

void CudaPICSimulator::ComputeExternalForces(double timeIntervalInSeconds)
{
    // TO DO
    const auto& size = _fluid.velocityGrid.GetSize();
    auto& velGrid = _fluid.velocityGrid;
    for(size_t forceIdx = 0; forceIdx < _externalForces.size(); forceIdx++)
    {
        _externalForces[forceIdx]->ApplyExternalForce(velGrid, timeIntervalInSeconds);
    }
    ApplyBoundryCondition();
}

void CudaPICSimulator::ComputeDiffusion(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _diffusionSolver->Solve(currentVelocity, _fluid.sdf, _boundryConditionSolver->GetColliderSdf(), timeIntervalInSeconds, _fluid.viscosity, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void CudaPICSimulator::ComputePressure(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _pressureSolver->Solve(currentVelocity, _fluid.sdf, _boundryConditionSolver->GetColliderSdf(), timeIntervalInSeconds, _fluid.density, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void CudaPICSimulator::ComputeAdvection(double timeIntervalInSeconds)
{
    ExtrapolateVelocityToAir();
    ApplyBoundryCondition();
    TransferGrid2Particles();
    MoveParticles(timeIntervalInSeconds);
}

void CudaPICSimulator::MoveParticles(double timeIntervalInSeconds)
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

        if (pt1.x <= _domain.GetOrigin().x) 
        {
            pt1.x = _domain.GetOrigin().x;
            vel.x = vel.x < 0.0 ? 0.0 : vel.x;
        }
        if (pt1.x >= _domain.GetOrigin().x + _domain.GetSize().x) 
        {
            pt1.x = _domain.GetOrigin().x + _domain.GetSize().x;
            vel.x = vel.x > 0.0 ? 0.0 : vel.x;
        }
        if (pt1.y <= _domain.GetOrigin().y) 
        {
            pt1.y = _domain.GetOrigin().y;
            vel.y = vel.y < 0.0 ? 0.0 : vel.y;
        }
        if (pt1.y >= _domain.GetOrigin().y + _domain.GetSize().y) 
        {
            pt1.y = _domain.GetOrigin().y + _domain.GetSize().y;
            vel.y = vel.y > 0.0 ? 0.0 : vel.y;
        }
        if (pt1.z <= _domain.GetOrigin().z) 
        {
            pt1.z = _domain.GetOrigin().z;
            vel.z = vel.z < 0.0 ? 0.0 : vel.z;
        }
        if (pt1.z >= _domain.GetOrigin().z + _domain.GetSize().z) 
        {
            pt1.z = _domain.GetOrigin().z + _domain.GetSize().z;
            vel.z = vel.z > 0.0 ? 0.0 : vel.z;
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

void CudaPICSimulator::ApplyBoundryCondition()
{
    _boundryConditionSolver->ConstrainVelocity(_fluid.velocityGrid, static_cast<size_t>(std::ceil(_maxClf)));
}

void CudaPICSimulator::BeginAdvanceTimeStep(double tmeIntervalInSecons)
{
    OnBeginAdvanceTimeStep(tmeIntervalInSecons);
}

void CudaPICSimulator::EndAdvanceTimeStep(double tmeIntervalInSecons)
{
    OnEndAdvanceTimeStep(tmeIntervalInSecons);
}

unsigned int CudaPICSimulator::NumberOfSubTimeSteps(double tmeIntervalInSecons) const
{
    double currentCfl = Cfl(tmeIntervalInSecons);
    std::cout << "Current CFL: " << currentCfl << "\n";
    unsigned int numberOfSubSteps = static_cast<unsigned int>(std::max(std::ceil(currentCfl / _maxClf), 1.0));
    return std::min(numberOfSubSteps, _maxNumberOfSubSteps);
}

void CudaPICSimulator::TransferParticles2Grid()
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

void CudaPICSimulator::TransferGrid2Particles()
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

__global__ void CUDA_BuildSignedDistanceFieldFromParticles(double* fluidSdf, CUDA_Vector3* particlePos, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, double radious, int numberOfParticles, CUDA_Int3 size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        double minDistance = radious;
        CUDA_Vector3 pos = CUDA_GridIdxToPosition({i, j, k}, gridSpacing, origin);
        CUDA_Vector3 lengthVector = {0, 0, 0};
        for(size_t particleIdx = 0; particleIdx < numberOfParticles; particleIdx++)
        {
            lengthVector = {particlePos[particleIdx].x - pos.x, particlePos[particleIdx].y - pos.y, particlePos[particleIdx].z - pos.z};
            minDistance = min(minDistance, CUDA_Vector3GetLength(lengthVector));
        }
        fluidSdf[idx] = minDistance - radious;
    }
}

void CudaPICSimulator::BuildSignedDistanceField()
{
    // TO DO CUDA
    auto& sdf = _fluid.sdf;
    auto cudaSize = Vector3SizeToCUDA_Int3(sdf.GetSize());
    auto& particleSystem = _fluid.particleSystem;
    int numberOfParticles = static_cast<int>(particleSystem.GetParticleNumber());
    auto gridSpacing = Vector3ToCUDA_Vector3(sdf.GetGridSpacing());
    auto dataOrigin = Vector3ToCUDA_Vector3(sdf.GetOrigin());
    double maxH = std::max(gridSpacing.x, std::max(gridSpacing.y, gridSpacing.z));
    double radious = 1.2 * maxH / std::sqrt(2.0);
    double sdfBandRadious = 2.0 * radious;

    int blocksInX = (int)std::ceil(((double)cudaSize.x) / _xThreads);
    int blocksInY = (int)std::ceil(((double)cudaSize.y) / _yThreads);
    int blocksInZ = (int)std::ceil(((double)cudaSize.z) / _zThreads);

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(_xThreads, _yThreads, _zThreads);

    CUDA_BuildSignedDistanceFieldFromParticles<<<dimGrid, dimBlock>>>(_d_fluidSdf, _d_particlesPosition, dataOrigin, gridSpacing, sdfBandRadious, numberOfParticles, cudaSize);

    // particleSystem.BuildSearcher(PARTICLE_POSITION_KEY, sdfBandRadious); 
    // auto searcher = particleSystem.GetSearcher();
    // sdf.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    // {
    //     Vector3<double> gridPosition = sdf.GridIndexToPosition(i, j, k);
    //     double minDistance = sdfBandRadious;
    //     searcher->ForEachNearbyPoint(gridPosition, sdfBandRadious, [&](size_t, const Vector3<double>& x)
    //     {
            
    //         minDistance = std::min(minDistance, Collisions::DistanceToPoint(gridPosition, x));
    //     });
    //     sdf(i, j, k) = minDistance - radious;
    // });
    //ExtrapolateIntoCollider();
}

void CudaPICSimulator::ExtrapolateVelocityToAir()
{
    // TO DO CUDA
    auto& markers = _fluid.markers;
    const auto& size = markers.GetSize();
    auto& xVel = _fluid.velocityGrid.GetDataXRef();
    auto& yVel = _fluid.velocityGrid.GetDataYRef();
    auto& zVel = _fluid.velocityGrid.GetDataZRef();
    const auto prevX(xVel);
    const auto prevY(yVel);
    const auto prevZ(zVel);
    const auto& xMarker = _fluid.xMarkers;
    const auto& yMarker = _fluid.yMarkers;
    const auto& zMarker = _fluid.zMarkers;

    size_t numberOfIterations = static_cast<size_t>(std::ceil(MaxCfl()));
    WrappedCuda_ExtrapolateToRegion(prevX, xMarker, numberOfIterations, xVel);
    WrappedCuda_ExtrapolateToRegion(prevY, yMarker, numberOfIterations, yVel);
    WrappedCuda_ExtrapolateToRegion(prevZ, zMarker, numberOfIterations, zVel);
}

void CudaPICSimulator::ExtrapolateIntoCollider()
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

void CudaPICSimulator::InitializeParticles()
{
    // TO DO CUDA
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

void CudaPICSimulator::ReAllocateCudaMemory()
{
    if(_wasCudaInitialized)
    {
        FreeCudaMemory();
    }
    AllocateCudaMemory();
}

void CudaPICSimulator::AllocateCudaMemory()
{
    const auto& gridSize = _fluid.velocityGrid.GetSize();
    const size_t gridVectorizedSize = gridSize.x * gridSize.y * gridSize.z;
    const size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();

    cudaMalloc((void**)&_d_xVelocity, gridVectorizedSize * sizeof(double));
    cudaMalloc((void**)&_d_yVelocity, gridVectorizedSize * sizeof(double));
    cudaMalloc((void**)&_d_zVelocity, gridVectorizedSize * sizeof(double));
    cudaMalloc((void**)&_d_fluidSdf, gridVectorizedSize * sizeof(double));
    cudaMalloc((void**)&_d_xMarkers, gridVectorizedSize * sizeof(int));
    cudaMalloc((void**)&_d_yMarkers, gridVectorizedSize * sizeof(int));
    cudaMalloc((void**)&_d_zMarkers, gridVectorizedSize * sizeof(int));
    cudaMalloc((void**)&_d_particlesPosition, numberOfParticles * sizeof(CUDA_Vector3));
    cudaMalloc((void**)&_d_particlesVelocity, numberOfParticles * sizeof(CUDA_Vector3));
    _h_xVelocity = (double*)malloc(gridVectorizedSize * sizeof(double));
    _h_yVelocity = (double*)malloc(gridVectorizedSize * sizeof(double));
    _h_zVelocity = (double*)malloc(gridVectorizedSize * sizeof(double));
    _h_fluidSdf = (double*)malloc(gridVectorizedSize * sizeof(double));
    _h_xMarkers = (int*)malloc(gridVectorizedSize * sizeof(int));
    _h_yMarkers = (int*)malloc(gridVectorizedSize * sizeof(int));
    _h_zMarkers = (int*)malloc(gridVectorizedSize * sizeof(int));
    _h_particlesPosition = (CUDA_Vector3*)malloc(gridVectorizedSize * sizeof(CUDA_Vector3));
    _h_particlesVelocity = (CUDA_Vector3*)malloc(gridVectorizedSize * sizeof(CUDA_Vector3));
    _wasCudaInitialized = true;
}

void CudaPICSimulator::FreeCudaMemory()
{
    cudaFree(_d_xVelocity);
    cudaFree(_d_yVelocity);
    cudaFree(_d_zVelocity);
    cudaFree(_d_fluidSdf);
    cudaFree(_d_xMarkers);
    cudaFree(_d_yMarkers);
    cudaFree(_d_zMarkers);
    cudaFree(_d_particlesPosition);
    cudaFree(_d_particlesVelocity);
    free(_h_xVelocity);
    free(_h_yVelocity);
    free(_h_zVelocity);
    free(_h_fluidSdf);
    free(_h_xMarkers);
    free(_h_yMarkers);
    free(_h_zMarkers);
    free(_h_particlesPosition);
    free(_h_particlesVelocity);
    
    _wasCudaInitialized = false;
}

void CudaPICSimulator::CopyFluidStateHostToDevice()
{
    CopyGridHostToDevice();
    CopyParticlesHostToDevice();
}   

void CudaPICSimulator::CopyFluidStateDeviceToHost()
{
    CopyGridDeviceToHost();
    CopyParticlesDeviceToHost();
}

void CudaPICSimulator::CopyParticlesHostToDevice()
{
    const auto& numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    const auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    const auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        _h_particlesPosition[i] = Vector3ToCUDA_Vector3(particlesPos[i]);
        _h_particlesVelocity[i] = Vector3ToCUDA_Vector3(particlesVel[i]);
    });

    cudaMemcpy(_d_particlesPosition, _h_particlesPosition, numberOfParticles * sizeof(CUDA_Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_particlesVelocity, _h_particlesVelocity, numberOfParticles * sizeof(CUDA_Vector3), cudaMemcpyHostToDevice);
}

void CudaPICSimulator::CopyParticlesDeviceToHost()
{
    const auto& numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    cudaMemcpy(_h_particlesPosition, _d_particlesPosition, numberOfParticles * sizeof(CUDA_Vector3), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_particlesVelocity, _d_particlesVelocity, numberOfParticles * sizeof(CUDA_Vector3), cudaMemcpyDeviceToHost);

    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        particlesPos[i] = CUDA_Vector3ToVector3(_h_particlesPosition[i]);
        particlesVel[i] = CUDA_Vector3ToVector3(_h_particlesVelocity[i]);
    });
}

void CudaPICSimulator::CopyGridHostToDevice()
{
    const auto& velocityGrid = _fluid.velocityGrid;
    const auto& size = velocityGrid.GetSize();
    const size_t gridVectorizedSize = size.x * size.y * size.z;

    velocityGrid.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        _h_xVelocity[i + size.x * (j + size.y * k)] = velocityGrid.x(i, j, k);
        _h_yVelocity[i + size.x * (j + size.y * k)] = velocityGrid.y(i, j, k);
        _h_zVelocity[i + size.x * (j + size.y * k)] = velocityGrid.z(i, j, k);
        _h_fluidSdf[i + size.x * (j + size.y * k)] = _fluid.sdf(i, j, k);
        _h_xMarkers[i + size.x * (j + size.y * k)] = _fluid.xMarkers(i, j, k);
        _h_yMarkers[i + size.x * (j + size.y * k)] = _fluid.yMarkers(i, j, k);
        _h_zMarkers[i + size.x * (j + size.y * k)] = _fluid.zMarkers(i, j, k);
    });

    cudaMemcpy(_d_xVelocity, _h_xVelocity, gridVectorizedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_yVelocity, _h_yVelocity, gridVectorizedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_zVelocity, _h_zVelocity, gridVectorizedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_fluidSdf, _h_fluidSdf, gridVectorizedSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_xMarkers, _h_xMarkers, gridVectorizedSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_yMarkers, _h_yMarkers, gridVectorizedSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_zMarkers, _h_zMarkers, gridVectorizedSize * sizeof(int), cudaMemcpyHostToDevice);
}

void CudaPICSimulator::CopyGridDeviceToHost()
{
    auto& velocityGrid = _fluid.velocityGrid;
    const auto& size = velocityGrid.GetSize();
    const size_t gridVectorizedSize = size.x * size.y * size.z;

    cudaMemcpy(_h_xVelocity, _d_xVelocity, gridVectorizedSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_yVelocity, _d_yVelocity, gridVectorizedSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_zVelocity, _d_zVelocity, gridVectorizedSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_fluidSdf, _d_fluidSdf, gridVectorizedSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_xMarkers, _d_xMarkers, gridVectorizedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_yMarkers, _d_yMarkers, gridVectorizedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(_h_zMarkers, _d_zMarkers, gridVectorizedSize * sizeof(int), cudaMemcpyDeviceToHost);

    velocityGrid.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        velocityGrid.x(i, j, k) = _h_xVelocity[i + size.x * (j + size.y * k)];
        velocityGrid.y(i, j, k) = _h_yVelocity[i + size.x * (j + size.y * k)];
        velocityGrid.z(i, j, k) = _h_zVelocity[i + size.x * (j + size.y * k)];
        _fluid.sdf(i, j, k) = _h_fluidSdf[i + size.x * (j + size.y * k)];
        _fluid.xMarkers(i, j, k) = _h_xMarkers[i + size.x * (j + size.y * k)];
        _fluid.yMarkers(i, j, k) = _h_yMarkers[i + size.x * (j + size.y * k)];
        _fluid.zMarkers(i, j, k) = _h_zMarkers[i + size.x * (j + size.y * k)];
    });
}

