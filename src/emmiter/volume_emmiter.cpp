#include "volume_emmiter.hpp"

#include <vector>
#include <random>

#include "../particle_systems/hash_grid_particle_searcher.hpp"
#include "../3d/collisions.hpp" 
#include "../3d/mesh_2_sdf.hpp"



VolumeEmitter::VolumeEmitter(const TriangleMesh& mesh, Vector3<size_t> resolution, BoundingBox3D domain, const size_t& particlesPerBlock, 
                             const Vector3<double>& velocity, const Vector3<double>& velocityVariance)
    : _startVelocity(velocity), _velocityVariance(velocityVariance)
{
    auto domainSize = domain.GetSize();
    auto gridSpacing = Vector3<double>(domainSize.x / resolution.x, domainSize.y / resolution.y, domainSize.z / resolution.z);
    auto origin = domain.GetOrigin();
    _sdf = ScalarGrid3D(resolution, 0.0, origin, gridSpacing);
    _particlesPerBlock = particlesPerBlock;
    InitializeFromTriangleMesh(mesh);
}


VolumeEmitter::~VolumeEmitter()
{

}

void VolumeEmitter::InitializeFromTriangleMesh(const TriangleMesh& mesh)
{
    Mesh2SDF converter;
    converter.Build(mesh, _sdf);
    // const auto& verticies = mesh.GetVerticies();
    // Vector3<double> origin;
    // Vector3<double> size;
    // for(size_t i = 0; i < verticies.size(); i++)
    // {
    //     origin.x = min(origin.x, verticies[i].x);
    //     origin.y = min(origin.y, verticies[i].y);
    //     origin.z = min(origin.z, verticies[i].z);
    //     size.x = max(size.x, verticies[i].x - origin.x);
    //     size.y = max(size.y, verticies[i].y - origin.y);
    //     size.z = max(size.z, verticies[i].z - origin.z);
    // }
    // _objectBBox.SetOrigin(origin - _sdf.GetGridSpacing());
    // _objectBBox.SetSize(size + _sdf.GetGridSpacing());
}

void VolumeEmitter::InitializeFromSdf(const ScalarGrid3D& sdf)
{
    _sdf.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        _sdf(i, j, k) = sdf(i, j, k);
    });
}

void VolumeEmitter::Emitt(ParticleSystem& particleSystem, std::string posKey, std::string velKey, const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf)
{
    std::vector<Vector3<double>> newPositions;

    const auto& gridSpacing = _sdf.GetGridSpacing();

    _sdf.ForEachIndex([&](size_t i, size_t j, size_t k)
    {
        const Vector3<double> pos = _sdf.GridIndexToPosition(i, j, k);
        if(_sdf(i, j, k) <= 0 && fluidSdf(i, j, k) >= 0 && colliderSdf(i, j, k) > 0)
        {
            for(size_t candidateIdx = 0; candidateIdx < _particlesPerBlock; candidateIdx++)
            {
                double x = gridSpacing.x * ( (double)std::rand() / (double)RAND_MAX ) + pos.x;
                double y = gridSpacing.y * ( (double)std::rand() / (double)RAND_MAX ) + pos.y;
                double z = gridSpacing.z * ( (double)std::rand() / (double)RAND_MAX ) + pos.z;
                newPositions.push_back(Vector3<double>(x, y, z));
            }
        }
    });

    size_t previousLastParticle = particleSystem.GetParticleNumber();
    particleSystem.AddParticles(newPositions.size(), newPositions, posKey);
    auto& velocities = particleSystem.GetVectorValues(velKey);
    for(size_t i = 0; i < newPositions.size(); i++)
    {
        velocities[previousLastParticle + i] = _startVelocity;
    }
}