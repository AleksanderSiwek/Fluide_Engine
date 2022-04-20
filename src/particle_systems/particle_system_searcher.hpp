#ifndef _PARTICLE_SYSTEM_SEARHER_HPP
#define _PARTICLE_SYSTEM_SEARHER_HPP

#include <vector>

#include "../common/vector3.hpp"


class ParticleSystemSearcher
{
    public:
        ParticleSystemSearcher() {}
        virtual ~ParticleSystemSearcher() {}

        virtual void Build(const std::vector<Vector3<double>>&points) = 0;
        virtual bool HasNearbyPoint(Vector3<double> position, double radious) = 0;
        virtual std::vector<size_t> GetNearbyPointsIndexes(Vector3<double> position, double radious) = 0;
};

#endif // _PARTICLE_SYSTEM_SEARHER_HPP