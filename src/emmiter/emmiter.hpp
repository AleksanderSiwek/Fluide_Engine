#ifndef _EMMITER_HPP
#define _EMMITER_HPP

#include <string>

#include "../particle_systems/particle_system.hpp"
#include "../3d/scalar_grid3d.hpp"


class Emitter
{
    public:
        Emitter();

        virtual ~Emitter();

        virtual void Emitt(ParticleSystem& particleSystem, std::string posKey, std::string velKey, const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf) = 0;
};

#endif // _EMMITER_HPP