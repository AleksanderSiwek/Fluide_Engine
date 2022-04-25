#include "pic_simulator.hpp"


PICSimulator::PICSimulator(const Vector3<size_t>& resolution, const Vector3<double> gridSpacing, const Vector3<double> gridOrigin)
{
    _fluid.velocityGrid.Resize(resolution);
    _fluid.velocityGrid.SetGridSpacing(gridSpacing);
    _fluid.velocityGrid.SetOrigin(gridOrigin);
    _fluid.markers.Resize(resolution);
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
    ComputeAdvection(timeIntervalInSeconds);

    EndAdvanceTimeStep(timeIntervalInSeconds);
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
    _pressureSolver->Solve(currentVelocity, FluidMarkers(_fluid.velocityGrid.GetSize()), timeIntervalInSeconds, _fluid.density, &(_fluid.velocityGrid));
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
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(0);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(1);
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
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(0);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(1);

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
    // TO DO: do this prettier cause tihs looks like ...
    particleSystem.BuildSearcher("position", 2 * radious); 
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

    size_t depth = static_cast<size_t>(std::ceil(_maxClf));
    ExtrapolateToRegion(sdf, markers, depth);
}