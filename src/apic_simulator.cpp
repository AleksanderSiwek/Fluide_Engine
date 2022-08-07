#include "apic_simulator.hpp"


APICSimulator::APICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
    : PICSimulator(gridSize, domain)
{

}

APICSimulator::~APICSimulator()
{
    
}

void APICSimulator::TransferParticles2Grid()
{
    auto& velocity = _fluid.velocityGrid;
    const auto& size = velocity.GetSize();
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& positions = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& velocities = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    _cX.resize(numberOfParticles);
    _cY.resize(numberOfParticles);
    _cZ.resize(numberOfParticles);

    velocity.ParallelFill(0, 0, 0);

    auto& u = velocity.GetDataXRef();
    auto& v = velocity.GetDataYRef();
    auto& w = velocity.GetDataZRef();

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

    const auto& hh = velocity.GetGridSpacing() / 2.0;
    const auto& domainOrigin = _domain.GetOrigin();
    const auto& domainSize = _domain.GetSize();

    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        std::array<Vector3<size_t>, 8> indices;
        std::array<double, 8> weights;

        auto uPosClamped = positions[i];
        uPosClamped.y = Clamp<double>(uPosClamped.y, domainOrigin.y + domainSize.y - hh.y, domainOrigin.y + hh.y);
        uPosClamped.z = Clamp<double>(uPosClamped.z, domainOrigin.z + domainSize.z - hh.z, domainOrigin.z + hh.z);
        GetCooridnatesAndWeights(size, velocity.GetDataXOrigin(), velocity.GetGridSpacing(), positions[i], indices, weights);
        for(size_t j = 0; j < 8; j++)
        {
            Vector3<double> gridPos = velocity.GetXPos(indices[j].x, indices[j].y, indices[j].z);
            double apicTerm = _cX[i].Dot(gridPos - uPosClamped);
            u(indices[j]) += (velocities[i].x + apicTerm) * weights[j];
            uWeight(indices[j]) += weights[j];
            xMarkers(indices[j]) = 1;
        }

        auto vPosClamped = positions[i];
        vPosClamped.x = Clamp<double>(vPosClamped.x, domainOrigin.x + domainSize.x - hh.x, domainOrigin.x + hh.x);
        vPosClamped.z = Clamp<double>(vPosClamped.z, domainOrigin.z + domainSize.z - hh.z, domainOrigin.z + hh.z);
        GetCooridnatesAndWeights(size, velocity.GetDataYOrigin(), velocity.GetGridSpacing(), positions[i], indices, weights);
        for(size_t j = 0; j < 8; j++)
        {
            Vector3<double> gridPos = velocity.GetYPos(indices[j].x, indices[j].y, indices[j].z);
            double apicTerm = _cY[i].Dot(gridPos - vPosClamped);
            v(indices[j]) += (velocities[i].y + apicTerm) * weights[j];
            vWeight(indices[j]) += weights[j];
            yMarkers(indices[j]) = 1;
        }

        auto wPosClamped = positions[i];
        wPosClamped.x = Clamp<double>(wPosClamped.x, domainOrigin.x + domainSize.x - hh.x, domainOrigin.x + hh.x);
        wPosClamped.y = Clamp<double>(wPosClamped.y, domainOrigin.y + domainSize.y - hh.y, domainOrigin.y + hh.y);
        GetCooridnatesAndWeights(size, velocity.GetDataZOrigin(), velocity.GetGridSpacing(), positions[i], indices, weights);
        for(size_t j = 0; j < 8; j++)
        {
            Vector3<double> gridPos = velocity.GetZPos(indices[j].x, indices[j].y, indices[j].z);
            double apicTerm = _cZ[i].Dot(gridPos - wPosClamped);
            w(indices[j]) += (velocities[i].z + apicTerm) * weights[j];
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

void APICSimulator::TransferGrid2Particles()
{
    const auto& velocity = _fluid.velocityGrid;
    const auto& size = velocity.GetSize();
    size_t numberOfParticles = _fluid.particleSystem.GetParticleNumber();
    auto& positions = _fluid.particleSystem.GetVectorValues(PARTICLE_POSITION_KEY);
    auto& velocities = _fluid.particleSystem.GetVectorValues(PARTICLE_VELOCITY_KEY);

    _cX.resize(numberOfParticles);
    _cY.resize(numberOfParticles);
    _cZ.resize(numberOfParticles);
    std::fill(_cX.begin(), _cX.end(), 0);
    std::fill(_cY.begin(), _cY.end(), 0);
    std::fill(_cZ.begin(), _cZ.end(), 0);

    const auto& u = velocity.GetDataXRef();
    const auto& v = velocity.GetDataYRef();
    const auto& w = velocity.GetDataZRef();

    const auto& hh = velocity.GetGridSpacing() / 2.0;
    const auto& domainOrigin = _domain.GetOrigin();
    const auto& domainSize = _domain.GetSize();

    _fluid.particleSystem.ParallelForEachParticle([&](size_t i)
    {
        velocities[i] = velocity.Sample(positions[i]);

        std::array<Vector3<size_t>, 8> indices;
        std::array<Vector3<double>, 8> weights;

        auto uPosClamped = positions[i];
        uPosClamped.y = Clamp<double>(uPosClamped.y, domainOrigin.y + domainSize.y - hh.y, domainOrigin.y + hh.y);
        uPosClamped.z = Clamp<double>(uPosClamped.z, domainOrigin.z + domainSize.z - hh.z, domainOrigin.z + hh.z);
        GetCooridnatesAndGradientWeights(size, velocity.GetDataXOrigin(), velocity.GetGridSpacing(), uPosClamped, indices, weights);
        for(size_t j = 0; j < 8; j++)
        {
            _cX[i] += weights[j] * u(indices[j]);
        }

        auto vPosClamped = positions[i];
        vPosClamped.x = Clamp<double>(vPosClamped.x, domainOrigin.x + domainSize.x - hh.x, domainOrigin.x + hh.x);
        vPosClamped.z = Clamp<double>(vPosClamped.z, domainOrigin.z + domainSize.z - hh.z, domainOrigin.z + hh.z);
        GetCooridnatesAndGradientWeights(size, velocity.GetDataYOrigin(), velocity.GetGridSpacing(), vPosClamped, indices, weights);
        for(size_t j = 0; j < 8; j++)
        {
            _cY[i] += weights[j] * v(indices[j]);
        }

        auto wPosClamped = positions[i];
        wPosClamped.x = Clamp<double>(wPosClamped.x, domainOrigin.x + domainSize.x - hh.x, domainOrigin.x + hh.x);
        wPosClamped.y = Clamp<double>(wPosClamped.y, domainOrigin.y + domainSize.y - hh.y, domainOrigin.y + hh.y);
        GetCooridnatesAndGradientWeights(size, velocity.GetDataZOrigin(), velocity.GetGridSpacing(), wPosClamped, indices, weights);
        for(size_t j = 0; j < 8; j++)
        {
            _cZ[i] += weights[j] * w(indices[j]);
        }
    });
}
