#include "./hash_grid_particle_searcher.hpp"


HashGridParticleSearcher::HashGridParticleSearcher(const Vector3<size_t> size, double gridSpacing) : _size(size), _gridSpacing(gridSpacing)
{

}

HashGridParticleSearcher::~HashGridParticleSearcher()
{

}

void HashGridParticleSearcher::Build(const std::vector<Vector3<double>>& points)
{
    size_t numberOfParticles = points.size();
    if(numberOfParticles == 0) return;

    ClearAll();
    AllocateMemory(numberOfParticles);

    for(size_t i = 0; i < points.size(); i++)
    {
        _points[i] = points[i];
        size_t key = GetHashKeyFromPosition(points[i]);
        _buckets[key].push_back(i);
        // Vector3<int> bucketIdx = GetBucketIndex(points[i]);
        // _buckets[GetHashKeyFromBucketIndex(bucketIdx)].push_back(i);
        // SetLowestBucketIdx(bucketIdx);
        // SetHighestBucketIdx(bucketIdx);
    }
}

void HashGridParticleSearcher::ClearAll()
{
    _points.clear();
    _buckets.clear();
}

void HashGridParticleSearcher::AllocateMemory(size_t numberOfParticles)
{
    _points.resize(numberOfParticles);
    _buckets.resize(_size.x * _size.y * _size.z);
}

bool HashGridParticleSearcher::HasNearbyPoint(Vector3<double> position, double radious)
{
    if(_buckets.empty()) return false;

    std::array<size_t, 8> nearbyKeys;
    GetNearbyKeys(position, nearbyKeys);

    const double radiousSquared = radious * radious;

    for(size_t i = 0; i < nearbyKeys.size(); i++)
    {
        const auto& bucket = _buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.size();
        for(size_t j = 0; j < numberOfPointsInBucket; j++)
        {
            size_t pointIdx = bucket[j];
            double distance = (_points[pointIdx] - position).GetLength();
            if(distance * distance <= radiousSquared) 
                return true;
        } 
    }

    return false;
}

std::vector<size_t> HashGridParticleSearcher::GetNearbyPointsIndexes(Vector3<double> position, double radious)
{
    std::vector<size_t> nearbyIndexes;
    if(_buckets.empty()) return nearbyIndexes;

    std::array<size_t, 8> nearbyKeys;
    GetNearbyKeys(position, nearbyKeys);

    const double radiousSquared = radious * radious;

    for(size_t i = 0; i < nearbyKeys.size(); i++)
    {
        const auto& bucket = _buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.size();
        for(size_t j = 0; j < numberOfPointsInBucket; j++)
        {
            size_t pointIdx = bucket[j];
            double distance = (_points[pointIdx] - position).GetLength();
            if(distance * distance <= radiousSquared) 
                nearbyIndexes.push_back(pointIdx);
        } 
    }
    
    return nearbyIndexes;
}

void HashGridParticleSearcher::ForEachNearbyPoint(const Vector3<double>& position, double radius, const ForEachPointCallback callback) const 
{
    if(_buckets.empty())
    {
        return;
    }

    std::array<size_t, 8> nearbyKeys;
    GetNearbyKeys(position, nearbyKeys);

    const double queryRadiusSquared = radius * radius;

    for (int i = 0; i < 8; i++) 
    {
        const auto& bucket = _buckets[nearbyKeys[i]];
        size_t numberOfPointsInBucket = bucket.size();

        for (size_t j = 0; j < numberOfPointsInBucket; ++j) 
        {
            size_t pointIndex = bucket[j];
            double distanceSquared = (_points[pointIndex] - position).GetLengthSquared();
            if (distanceSquared <= queryRadiusSquared) 
            {
                callback(pointIndex, _points[pointIndex]);
            }
        }

    }

}

size_t HashGridParticleSearcher::GetHashKeyFromPosition(const Vector3<double>& position) const
{
    return GetHashKeyFromBucketIndex(GetBucketIndex(position));
}

Vector3<int> HashGridParticleSearcher::GetBucketIndex(const Vector3<double>& position) const
{
    Vector3<int> bucketIdx;
    bucketIdx.x = static_cast<int>(std::floor(position.x / _gridSpacing));
    bucketIdx.y = static_cast<int>(std::floor(position.y / _gridSpacing));
    bucketIdx.z = static_cast<int>(std::floor(position.z / _gridSpacing));
    return bucketIdx;
}

size_t HashGridParticleSearcher::GetHashKeyFromBucketIndex(const Vector3<int>& bucketIndex) const
{
    Vector3<int> wrappedIdx;
    wrappedIdx.x = bucketIndex.x % _size.x;
    wrappedIdx.y = bucketIndex.y % _size.y;
    wrappedIdx.z = bucketIndex.z % _size.z;
    if(wrappedIdx.x < 0) wrappedIdx.x += static_cast<int>(_size.x);
    if(wrappedIdx.y < 0) wrappedIdx.y += static_cast<int>(_size.y);
    if(wrappedIdx.z < 0) wrappedIdx.z += static_cast<int>(_size.z);
    return static_cast<size_t>((wrappedIdx.z * _size.y + wrappedIdx.y) * _size.x + wrappedIdx.x);
}

void HashGridParticleSearcher::GetNearbyKeys(const Vector3<double>& position, std::array<size_t, 8>& nearbyKeys) const
{
    Vector3<int> originIndex = GetBucketIndex(position);
    Vector3<int> nearbyBucketIndices[8];

     for (int i = 0; i < 8; i++) 
    {
        nearbyBucketIndices[i] = originIndex;
    }

    if ((originIndex.x + 0.5f) * _gridSpacing <= position.x) {
        nearbyBucketIndices[4].x += 1;
        nearbyBucketIndices[5].x += 1;
        nearbyBucketIndices[6].x += 1;
        nearbyBucketIndices[7].x += 1;
    } else {
        nearbyBucketIndices[4].x -= 1;
        nearbyBucketIndices[5].x -= 1;
        nearbyBucketIndices[6].x -= 1;
        nearbyBucketIndices[7].x -= 1;
    }

    if ((originIndex.y + 0.5f) * _gridSpacing <= position.y) {
        nearbyBucketIndices[2].y += 1;
        nearbyBucketIndices[3].y += 1;
        nearbyBucketIndices[6].y += 1;
        nearbyBucketIndices[7].y += 1;
    } else {
        nearbyBucketIndices[2].y -= 1;
        nearbyBucketIndices[3].y -= 1;
        nearbyBucketIndices[6].y -= 1;
        nearbyBucketIndices[7].y -= 1;
    }

    if ((originIndex.z + 0.5f) * _gridSpacing <= position.z) {
        nearbyBucketIndices[1].z += 1;
        nearbyBucketIndices[3].z += 1;
        nearbyBucketIndices[5].z += 1;
        nearbyBucketIndices[7].z += 1;
    } else {
        nearbyBucketIndices[1].z -= 1;
        nearbyBucketIndices[3].z -= 1;
        nearbyBucketIndices[5].z -= 1;
        nearbyBucketIndices[7].z -= 1;
    }

    for (int i = 0; i < 8; i++) 
    {
        nearbyKeys[i] = GetHashKeyFromBucketIndex(nearbyBucketIndices[i]);
    }
}

void HashGridParticleSearcher::SetLowestBucketIdx(const Vector3<int>& bucketIdx)
{
    if(bucketIdx.x < _lowestBucketIdx.x && bucketIdx.y < _lowestBucketIdx.y && bucketIdx.z < _lowestBucketIdx.z)
        _lowestBucketIdx = bucketIdx;
}

void HashGridParticleSearcher::SetHighestBucketIdx(const Vector3<int>& bucketIdx)
{
    if(bucketIdx.x > _highestBucketIdx.x && bucketIdx.y > _highestBucketIdx.y && bucketIdx.z > _highestBucketIdx.z)
        _highestBucketIdx = bucketIdx;
}

bool HashGridParticleSearcher::IsInBucketRange(const Vector3<int>& bucketIdx) const
{
    if(bucketIdx.x >= _lowestBucketIdx.x && bucketIdx.x <= _highestBucketIdx.x &&
       bucketIdx.y >= _lowestBucketIdx.y && bucketIdx.y <= _highestBucketIdx.y &&
       bucketIdx.z >= _lowestBucketIdx.z && bucketIdx.z <= _highestBucketIdx.z)
       return true;
    return false;
}