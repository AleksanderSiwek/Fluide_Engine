#ifndef _VOLUME_EMMITER_HPP
#define _VOLUME_EMMITER_HPP

#include <string>

#include "emmiter.hpp"

#include "../3d/triangle_mesh.hpp"
#include "../3d/scalar_grid3d.hpp"
#include "../3d/bounding_box_3d.hpp"


class VolumeEmitter : public Emitter
{
    public:
        VolumeEmitter(const TriangleMesh& mesh, Vector3<size_t> resolution, BoundingBox3D domain, const size_t& particlesPerBlock, 
                             const Vector3<double>& velocity, const Vector3<double>& velocityVariance);

        ~VolumeEmitter();

        void InitializeFromTriangleMesh(const TriangleMesh& mesh);
        void VolumeEmitter::InitializeFromSdf(const ScalarGrid3D& sdf);

        void Emitt(ParticleSystem& particleSystem, std::string posKey, std::string velKey, const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf) override;

    private:
        ScalarGrid3D _sdf;
        BoundingBox3D _objectBBox;
        Vector3<double> _startVelocity;
        Vector3<double> _velocityVariance;
        size_t _particlesPerBlock = 8;
};

#endif // _VOLUME_EMMITER_HPP