#ifndef _HASH_GRID_PARTICLE_SEARCHER_HPP
#define _HASH_GRID_PARTICLE_SEARCHER_HPP

#include <algorithm>
#include <array>

#include "particle_system_searcher.hpp"
#include "../common/grid3d.hpp"

// TO DO: fix for speedup

class HashGridParticleSearcher : public ParticleSystemSearcher
{
    public:
        HashGridParticleSearcher(const Vector3<size_t> size, double gridSpacing);

        ~HashGridParticleSearcher();

        void Build(const std::vector<Vector3<double>>& points) override;
        bool HasNearbyPoint(Vector3<double> position, double radious) override;
        std::vector<size_t> GetNearbyPointsIndexes(Vector3<double> position, double radious) override;
        void ForEachNearbyPoint(const Vector3<double>& position, double radius, const ForEachPointCallback callback) const override;

        size_t GetHashKeyFromPosition(const Vector3<double>& position) const;
        Vector3<int> GetBucketIndex(const Vector3<double>& position) const;
        size_t GetHashKeyFromBucketIndex(const Vector3<int>& bucketIndex) const;
        void GetNearbyKeys(const Vector3<double>& position, std::array<size_t, 8>& nearbyKeys) const;

    private:
        Vector3<size_t> _size;
        double _gridSpacing;
        std::vector<Vector3<double>> _points;
        std::vector<std::vector<size_t>> _buckets;
        Vector3<int> _lowestBucketIdx=0;
        Vector3<int> _highestBucketIdx=0;

        void ClearAll();
        void AllocateMemory(size_t numberOfParticles);
        void SetLowestBucketIdx(const Vector3<int>& bucketIdx);
        void SetHighestBucketIdx(const Vector3<int>& bucketIdx);
        bool IsInBucketRange(const Vector3<int>& bucketIdx) const;
};

#endif // _HASH_GRID_PARTICLE_SEARCHER_HPP