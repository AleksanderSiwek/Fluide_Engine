#include "flip_simulator.hpp"


FLIPSimulator::FLIPSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
    : PICSimulator(gridSize, domain), _velocityResiduals(_fluid.velocityGrid.GetSize(), _fluid.velocityGrid.GetOrigin(), _fluid.velocityGrid.GetGridSpacing()), _blendingFactor(0.05)
{

}

FLIPSimulator::~FLIPSimulator()
{

}

void FLIPSimulator::SetBlendingFactor(double factor)
{
    _blendingFactor = factor;
}

double FLIPSimulator::GetBlendingFactor() const
{
    return _blendingFactor;
}

void FLIPSimulator::TransferParticles2Grid()
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

    u.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        _velocityResiduals.x(i, j, k) = u(i, j, k);
    });
    v.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        _velocityResiduals.y(i, j, k) = v(i, j, k);
    });
    w.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        _velocityResiduals.z(i, j, k) = w(i, j, k);
    });
}

void FLIPSimulator::TransferGrid2Particles()
{
    auto& velocity = _fluid.velocityGrid;
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& particlesPos = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& particlesVel = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    auto& xData = velocity.GetDataXRef();
    auto& yData = velocity.GetDataYRef();
    auto& zData = velocity.GetDataZRef();

    // _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    // {
    //     particlesVel[i] = velocity.Sample(particlesPos[i]);
    // });
    // 
    
    xData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        _velocityResiduals.x(i, j, k) = xData(i, j, k) - _velocityResiduals.x(i, j, k);
    });

    yData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        _velocityResiduals.y(i, j, k) = yData(i, j, k) - _velocityResiduals.y(i, j, k);
    });
    zData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
     _velocityResiduals.z(i, j, k) = zData(i, j, k) - _velocityResiduals.z(i, j, k);
    });

    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        Vector3<double> flipVelocity = particlesVel[i] + _velocityResiduals.Sample(particlesPos[i]);
        if(_blendingFactor > 0.0)
        {
            Vector3<double> picVelocity = velocity.Sample(particlesPos[i]);
            flipVelocity = Lerp<Vector3<double>, double>(flipVelocity, picVelocity, _blendingFactor);
        }
        particlesVel[i] = flipVelocity;
    });
}