#ifndef _HASH_GRID_PARTICLE_SEARCHER_HPP
#define _HASH_GRID_PARTICLE_SEARCHER_HPP

#include <algorithm>

#include "particle_system_searcher.hpp"
#include "../common/grid3d.hpp"


class HashGridParticleSearcher : public ParticleSystemSearcher
{
    public:
        HashGridParticleSearcher(const Vector3<size_t> size, double gridSpacing);

        ~HashGridParticleSearcher();

        void build(const std::vector<Vector3<double>>& points) override;
        bool HasNearbyPoint(Vector3<double> position, double radious) override;
        std::vector<size_t> GetNearbyPointsIndexes(Vector3<double> position, double radious) override;

        size_t GetHashKeyFromPosition(const Vector3<double>& position) const;
        Vector3<size_t> GetBucketIndex(const Vector3<double>& position) const;
        size_t GetHashKeyFromBucketIndex(const Vector3<size_t>& bucketIndex) const;
        void GetNearbyKeys(const Vector3<double>& position, size_t* nearbyKeys) const;

    private:
        Vector3<size_t> _size;
        double _gridSpacing;
        std::vector<Vector3<double>> _points;
        std::vector<size_t> _keys;
        std::vector<size_t> _startIndexTable;
        std::vector<size_t> _endIndexTable;
        std::vector<size_t> _sortedIndexes;

        void ClearAll();
        void AllocateMemory(size_t numberOfParticles);
};

#endif // _HASH_GRID_PARTICLE_SEARCHER_HPP