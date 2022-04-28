#include "pic_simulator.hpp"

#include <random>

// TO DO: DELETE
#include <iostream>
#include <iomanip>

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
    _fluid.sdf.SetOrigin(domain.GetOrigin());
    _fluid.markers.Resize(gridSize);
    _fluid.particleSystem.AddVectorValue(PARTICLE_POSITION_KEY);
    _fluid.particleSystem.AddVectorValue(PARTICLE_VELOCITY_KEY);
    _fluid.viscosity = 0.5;
    _fluid.density = 1;

    _maxClf = 5.0;
    _particlesPerBlok = 32;

    SetDiffusionSolver(std::make_shared<BackwardEulerDiffusionSolver>());
    SetPressureSolver(std::make_shared<SinglePhasePressureSolver>());
}

PICSimulator::~PICSimulator()
{

}

FluidMarkers PICSimulator::GetMarkers() const
{
    FluidMarkers markers;
    const auto& size = _fluid.sdf.GetSize();
    markers.Resize(size);
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                if(_fluid.sdf(i, j, k) < 0)
                {
                    markers(i, j, k) = FLUID_MARK;
                }
                else
                {
                    markers(i, j, k) = AIR_MARK;
                }
            }
        }
    }
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
    //PrintGrid(_fluid.velocityGrid.GetDataYRef());

    const auto& positions = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    const auto& velocities = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);
    std::cout << "Particle position before move [0] = " << positions[0].x << ", " << positions[0].y << ", " << positions[0].z << "\n";
    std::cout << "Particle velocities before move [0] = " << velocities[0].x << ", " << velocities[0].y << ", " << velocities[0].z << "\n\n";

    ComputeExternalForces(timeIntervalInSeconds);
    //PrintGrid(_fluid.velocityGrid.GetDataYRef());

    ComputeDiffusion(timeIntervalInSeconds);
    // PrintGrid(_fluid.velocityGrid.GetDataYRef());

    ComputePressure(timeIntervalInSeconds);
    // PrintGrid(_fluid.velocityGrid.GetDataYRef());

    ComputeAdvection(timeIntervalInSeconds);
    //PrintGrid(_fluid.velocityGrid.GetDataYRef());

    const auto& positions1 = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    const auto& velocities1 = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);
    std::cout << "Particle position after move [0] = " << positions1[0].x << ", " << positions1[0].y << ", " << positions1[0].z << "\n";
    std::cout << "Particle velocities before move [0] = " << velocities1[0].x << ", " << velocities1[0].y << ", " << velocities1[0].z << "\n";

    EndAdvanceTimeStep(timeIntervalInSeconds);

    //PrintGrid(_fluid.velocityGrid.GetDataYRef());
}

void PICSimulator::OnBeginAdvanceTimeStep(double timeIntervalInSeconds)
{
    TransferParticles2Grid();
    BuildSignedDistanceField();
    ExtrapolateVelocityToAir();
    ApplyBoundryCondition();
}

void PICSimulator::OnEndAdvanceTimeStep(double timeIntervalInSeconds)
{

}

// TO DO: do it for custom forces
void PICSimulator::ComputeExternalForces(double timeIntervalInSeconds)
{
    const auto& size = _fluid.velocityGrid.GetSize();
    auto& velGrid = _fluid.velocityGrid;
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                velGrid.x(i, j, k) += timeIntervalInSeconds * 0;
                velGrid.y(i, j, k) += timeIntervalInSeconds * (-9.81);
                velGrid.z(i, j, k) += timeIntervalInSeconds * 0;
                ApplyBoundryCondition();
            }
        }
    }
}

void PICSimulator::ComputeDiffusion(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _diffusionSolver->Solve(currentVelocity, _fluid.sdf, timeIntervalInSeconds, _fluid.viscosity, &(_fluid.velocityGrid));
    ApplyBoundryCondition();
}

void PICSimulator::ComputePressure(double timeIntervalInSeconds)
{
    FaceCenteredGrid3D currentVelocity(_fluid.velocityGrid);
    _pressureSolver->Solve(currentVelocity, _fluid.sdf, timeIntervalInSeconds, _fluid.density, &(_fluid.velocityGrid));
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
    auto& velGrid = _fluid.velocityGrid;
    BoundingBox3D fluidBBox = _domain;
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    for(size_t i = 0; i < numberOfParticles; i++)
    {
        Vector3<double> pPos0 = particlesPos[i];
        Vector3<double> pPos1 = pPos0;
        Vector3<double> pVel1 = particlesVel[i];

        unsigned int numSubSteps = static_cast<unsigned int>(std::max(MaxCfl(), 1.0));
        double dt = timeIntervalInSeconds / numSubSteps;
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
            pVel1.x = std::min(std::abs(pVel1.x), _maxClf) * (std::abs(pVel1.x)/pVel1.x) * (-1);
        }

        if(pPos1.x >= fluidBBox.GetOrigin().x + fluidBBox.GetSize().x)
        {
            pPos1.x = fluidBBox.GetOrigin().x + fluidBBox.GetSize().x;
            pVel1.x = std::min(std::abs(pVel1.x), _maxClf) * (std::abs(pVel1.x)/pVel1.x) * (-1);
        }

        if(pPos1.y <= fluidBBox.GetOrigin().y)
        {
            pPos1.y = fluidBBox.GetOrigin().y;
            pVel1.y = std::min(std::abs(pVel1.y), _maxClf) * (std::abs(pVel1.y)/pVel1.y) * (-1);
        }

        if(pPos1.y >= fluidBBox.GetOrigin().y + fluidBBox.GetSize().y)
        {
            pPos1.y = fluidBBox.GetOrigin().y + fluidBBox.GetSize().y;
            pVel1.y = std::min(std::abs(pVel1.y), _maxClf) * (std::abs(pVel1.y)/pVel1.y) * (-1);
        }

        if(pPos1.z <= fluidBBox.GetOrigin().z)
        {
            pPos1.z = fluidBBox.GetOrigin().z;
            pVel1.z = std::min(std::abs(pVel1.z), _maxClf) * (std::abs(pVel1.z)/pVel1.z) * (-1);
        }

        if(pPos1.z >= fluidBBox.GetOrigin().z + fluidBBox.GetSize().z)
        {
            pPos1.z = fluidBBox.GetOrigin().z + fluidBBox.GetSize().z;
            pVel1.z = std::min(std::abs(pVel1.z), _maxClf) * (std::abs(pVel1.z)/pVel1.z) * (-1);
        }
    
        particlesPos[i] = pPos1;
        particlesVel[i] = pVel1;
    }
}

void PICSimulator::ApplyBoundryCondition()
{
    // TO DO
    // It works only for colliders, thats why its skipped right now
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
    auto& velGrid = _fluid.velocityGrid;
    const auto& size = velGrid.GetSize();
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);
    auto& markers = _fluid.markers;
    Array3<double> weightsArray(size, 0);

    velGrid.Fill(0, 0, 0);
    markers.Resize(size);

    Vector3<size_t> gridIndexes[8];
    double weights[8];
    for(size_t particleCnt = 0; particleCnt < numberOfParticles; particleCnt++)
    {
        Vector3<size_t> gridCellIndex = velGrid.PositionToGridIndex(particlesPos[particleCnt]);
        Vector3<double> gridCellPosition = velGrid.GridIndexToPosition(gridCellIndex);
        Vector3<double> pToCellPos = (particlesPos[particleCnt] - gridCellPosition) / velGrid.GetGridSpacing();

        gridIndexes[0] = Vector3<size_t>(gridCellIndex.x,     gridCellIndex.y,     gridCellIndex.z);
        gridIndexes[1] = Vector3<size_t>(gridCellIndex.x + 1, gridCellIndex.y,     gridCellIndex.z);
        gridIndexes[2] = Vector3<size_t>(gridCellIndex.x,     gridCellIndex.y + 1, gridCellIndex.z);
        gridIndexes[3] = Vector3<size_t>(gridCellIndex.x + 1, gridCellIndex.y + 1, gridCellIndex.z);
        gridIndexes[4] = Vector3<size_t>(gridCellIndex.x,     gridCellIndex.y,     gridCellIndex.z + 1);
        gridIndexes[5] = Vector3<size_t>(gridCellIndex.x + 1, gridCellIndex.y,     gridCellIndex.z + 1);
        gridIndexes[6] = Vector3<size_t>(gridCellIndex.x,     gridCellIndex.y + 1, gridCellIndex.z + 1);
        gridIndexes[7] = Vector3<size_t>(gridCellIndex.x + 1, gridCellIndex.y + 1, gridCellIndex.z + 1);

        weights[0] = (1.00 - pToCellPos.x) * (1.00 - pToCellPos.y) * (1.00 - pToCellPos.z);
        weights[1] = pToCellPos.x * (1.00 - pToCellPos.y) * (1.00 - pToCellPos.z);
        weights[2] = (1.00 - pToCellPos.x) * pToCellPos.y * (1.00 - pToCellPos.z);
        weights[3] = pToCellPos.x * pToCellPos.y * (1.00 - pToCellPos.z);
        weights[4] = (1.00 - pToCellPos.x) * (1.00 - pToCellPos.y) * pToCellPos.z;
        weights[5] = pToCellPos.x * (1.00 - pToCellPos.y) * pToCellPos.z;
        weights[6] = (1.00 - pToCellPos.x) * pToCellPos.y * pToCellPos.z; 
        weights[7] = pToCellPos.x * pToCellPos.y * pToCellPos.z; 

        for(size_t i = 0; i < 8; i++)
        {
            Vector3<size_t> idx = gridIndexes[i];
            if(idx.x < 0 || idx.y < 0 || idx.z < 0 ||idx.x >= size.x || idx.y >= size.y || idx.z >= size.z)
                continue;
            velGrid.x(idx.x ,idx.y, idx.z) += particlesVel[particleCnt].x * weights[i];
            velGrid.y(idx.x ,idx.y, idx.z) += particlesVel[particleCnt].y * weights[i];
            velGrid.z(idx.x ,idx.y, idx.z) += particlesVel[particleCnt].z * weights[i];  
            markers(idx) = FLUID_MARK;

            weightsArray(idx) += weights[i];         
        }
    }  

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                if(weightsArray(i, j, k) > 0.0)
                {
                    velGrid.x(i, j, k) /= weightsArray(i, j, k);
                    velGrid.y(i, j, k) /= weightsArray(i, j, k);
                    velGrid.z(i, j, k) /= weightsArray(i, j, k);  
                }
            }
        }
    } 

}

void PICSimulator::TransferGrid2Particles()
{
    auto& velGrid = _fluid.velocityGrid;
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    for(size_t i = 0; i < numberOfParticles; i++)
    {
        particlesVel[i] = velGrid.Sample(particlesPos[i]);
    }
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

    for(size_t i = 0; i < sdfSize.x; i++)
    {
        for(size_t j = 0;  j < sdfSize.y; j++)
        {
            for(size_t k = 0; k < sdfSize.z ; k++)
            {
                Vector3<double> gridPosition = sdf.GridIndexToPosition(i, j, k);
                double minDistance = sdfBandRadious;
                searcher->ForEachNearbyPoint(gridPosition, sdfBandRadious, [&](size_t, const Vector3<double>& x)
                {
                    minDistance = std::min(minDistance, Collisions::DistanceToPoint(x, gridPosition));
                });
                sdf(i, j, k) = minDistance - radious;
            }
        }
    }
    ExtrapolateIntoCollieder();
}

void PICSimulator::ExtrapolateVelocityToAir()
{
    auto& markers = _fluid.markers;
    auto& xVel = _fluid.velocityGrid.GetDataXRef();
    auto& yVel = _fluid.velocityGrid.GetDataYRef();
    auto& zVel = _fluid.velocityGrid.GetDataZRef();

    size_t numberOfIterations = static_cast<size_t>(std::ceil(MaxCfl()));
    ExtrapolateToRegion(xVel, markers, numberOfIterations);
    ExtrapolateToRegion(yVel, markers, numberOfIterations);
    ExtrapolateToRegion(zVel, markers, numberOfIterations);
}

void PICSimulator::ExtrapolateToRegion(Array3<double>& array, const FluidMarkers& valid, size_t numberOfIterations)
{
    const auto& size = array.GetSize();
    FluidMarkers valid0(valid);
    FluidMarkers valid1(valid);
    
    for(size_t iter = 0; iter < numberOfIterations; iter++)
    {
        for(int i = 0; i < size.x; i++)
        {
            for(int j = 0; j < size.y; j++)
            {
                for(int k = 0; k < size.z; k++)
                {
                    double sum = 0;
                    size_t cnt = 0;

                    if(valid0(i, j, k) != FLUID_MARK)
                    {
                        if(i + 1 < size.x && valid0(i + 1, j, k) == FLUID_MARK)
                        {
                            sum += array(i + 1, j, k);
                            cnt++;
                        }
                        
                        if(i - 1 >= 0 && valid0(i - 1, j, k) == FLUID_MARK)
                        {
                            sum += array(i - 1, j, k);
                            cnt++;
                        }

                        if(j + 1 < size.y && valid0(i, j + 1, k) == FLUID_MARK)
                        {
                            sum += array(i, j + 1, k);
                            cnt++;
                        }

                        if(j - 1 >= 0 && valid0(i, j - 1, k) == FLUID_MARK)
                        {
                            sum += array(i, j - 1, k);
                            cnt++;
                        }

                        if(k + 1 < size.z && valid0(i, j, k + 1) == FLUID_MARK)
                        {
                            sum += array(i, j, k + 1);
                            cnt++;
                        }

                        if(k - 1 >= 0 && valid0(i, j, k - 1) == FLUID_MARK)
                        {
                            sum += array(i, j, k - 1);
                            cnt++;
                        }

                        if(cnt > 0)
                        {
                            array(i, j, k) = sum / cnt;
                            valid1(i, j, k) = FLUID_MARK;
                        }
                    }
                }
            }
        }
        valid0.Fill(valid1);
    }
}

void PICSimulator::ExtrapolateIntoCollieder()
{
    auto& sdf = _fluid.sdf;
    const auto& size = sdf.GetSize();
    FluidMarkers markers(size);
    for(int i = 0; i < size.x; i++)
    {
        for(int j = 0; j < size.y; j++)
        {
            for(int k = 0; k < size.z; k++)
            {
                if(sdf(i, j, k) < 0)
                {
                    markers(i, j, k) = FLUID_MARK;
                }
                else
                {
                    markers(i, j, k) = AIR_MARK;
                }
            }
        }
    }

    // size_t depth = static_cast<size_t>(std::ceil(_maxClf));
    // ExtrapolateToRegion(sdf, markers, depth);
}

void PICSimulator::InitializeParticles()
{
    const auto& size = _fluid.sdf.GetSize();
    const auto& sdf = _fluid.sdf;
    const auto& velGrid = _fluid.velocityGrid;
    const auto& gridSpacing = velGrid.GetGridSpacing();
    auto& particles = _fluid.particleSystem;

    std::vector<Vector3<double>> positions;

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
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
            }
        }
    }

    particles.AddParticles(positions.size(), positions, PARTICLE_POSITION_KEY);
}