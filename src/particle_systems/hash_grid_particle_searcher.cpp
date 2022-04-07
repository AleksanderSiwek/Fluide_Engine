#include "./hash_grid_particle_searcher.hpp"


HashGridParticleSearcher::HashGridParticleSearcher(const Vector3<size_t> size, double gridSpacing) : _size(size), _gridSpacing(gridSpacing)
{

}

HashGridParticleSearcher::~HashGridParticleSearcher()
{

}

void HashGridParticleSearcher::build(const std::vector<Vector3<double>>& points)
{
    size_t numberOfParticles = points.size();
    if(numberOfParticles == 0) return;

    std::vector<size_t> tmpKeys(numberOfParticles);
    ClearAll();
    AllocateMemory(numberOfParticles);

    for(size_t i = 0; i < numberOfParticles; i++)
    {
        _sortedIndexes[i] = i;
        _points[i] = points[i];
        tmpKeys[i] = GetHashKeyFromPosition(points[i]);
    }

    std::sort(_sortedIndexes.begin(), _sortedIndexes.end(), [&tmpKeys](size_t indexA, size_t indexB)
    {
        return tmpKeys[indexA] < tmpKeys[indexB];
    });

    for(size_t i = 0; i < numberOfParticles; i++)
    {
        _points[i] = points[_sortedIndexes[i]];
        _keys[i] = tmpKeys[_sortedIndexes[i]];
    }

    _startIndexTable[_keys[0]] = 0;
    _endIndexTable[_keys[numberOfParticles - 1]] = numberOfParticles;

    for(size_t i = 1; i < numberOfParticles; i++)
    {
        if(_keys[i] > _keys[i - 1])
        {
            _startIndexTable[_keys[i]] = i;
            _endIndexTable[_keys[i -  1]] = i;
        }
    }
}

void HashGridParticleSearcher::ClearAll()
{
   _points.clear();
   _keys.clear();
   _startIndexTable.clear();
   _endIndexTable.clear();
   _sortedIndexes.clear(); 
}

void HashGridParticleSearcher::AllocateMemory(size_t numberOfParticles)
{
    _startIndexTable.resize(_size.x * _size.y * _size.z);
    _endIndexTable.resize(_size.x * _size.y * _size.z);
    _keys.resize(numberOfParticles);
    _sortedIndexes.resize(numberOfParticles);
    _points.resize(numberOfParticles);
}

bool HashGridParticleSearcher::HasNearbyPoint(Vector3<double> position, double radious)
{
    size_t nearbyKeys[8];
    GetNearbyKeys(position, nearbyKeys);
    const double radiousSq = radious * radious;

    for(size_t i = 0; i < 8; i++)
    {
        size_t nearbyKey = nearbyKeys[i];
        size_t start = _startIndexTable[nearbyKey];
        size_t end = _endIndexTable[nearbyKey];

        if(start == std::numeric_limits<double>::max()) continue;;

        for(size_t j = start; j < end; j++)
        {
            double distance = (_points[j] - position).GetLength();
            if(distance * distance < radiousSq)
                return true;
        }
    }

    return false;
}

std::vector<size_t> HashGridParticleSearcher::GetNearbyPointsIndexes(Vector3<double> position, double radious)
{
    size_t nearbyKeys[8];
    GetNearbyKeys(position, nearbyKeys);
    const double radiousSq = radious * radious;
    std::vector<size_t> nearbyIndexes;

    for(size_t i = 0; i < 8; i++)
    {
        size_t nearbyKey = nearbyKeys[i];
        size_t start = _startIndexTable[nearbyKey];
        size_t end = _endIndexTable[nearbyKey];

        if(start == std::numeric_limits<double>::max()) continue;;

        for(size_t j = start; j < end; j++)
        {
            double distance = (_points[j] - position).GetLength();
            if(distance * distance < radiousSq)
                nearbyIndexes.push_back(j);
        }
    }
    
    return nearbyIndexes;
}

size_t HashGridParticleSearcher::GetHashKeyFromPosition(const Vector3<double>& position) const
{
    return GetHashKeyFromBucketIndex(GetBucketIndex(position));
}

Vector3<size_t> HashGridParticleSearcher::GetBucketIndex(const Vector3<double>& position) const
{
    Vector3<size_t> bucketIdx;
    bucketIdx.x = static_cast<size_t>(std::floor(position.x / _gridSpacing));
    bucketIdx.y = static_cast<size_t>(std::floor(position.y / _gridSpacing));
    bucketIdx.z = static_cast<size_t>(std::floor(position.z / _gridSpacing));
    return bucketIdx;
}

size_t HashGridParticleSearcher::GetHashKeyFromBucketIndex(const Vector3<size_t>& bucketIndex) const
{
    Vector3<size_t> wrappedIdx;
    wrappedIdx.x = bucketIndex.x % _size.x;
    wrappedIdx.y = bucketIndex.y % _size.y;
    wrappedIdx.z = bucketIndex.z % _size.z;
    if(wrappedIdx.x < 0) wrappedIdx.x += _size.x;
    if(wrappedIdx.y < 0) wrappedIdx.y += _size.y;
    if(wrappedIdx.z < 0) wrappedIdx.z += _size.z;
    return static_cast<size_t>(((wrappedIdx).z * _size.y + wrappedIdx.y) * _size.x + wrappedIdx.x);
}

void HashGridParticleSearcher::GetNearbyKeys(const Vector3<double>& position, size_t* nearbyKeys) const
{
    Vector3<size_t> originIndex = GetBucketIndex(position);
    Vector3<size_t> nearbyBucketInexes[8];

    for(size_t i = 0; i < 8; i++)
    {
        nearbyBucketInexes[i] = originIndex;
    }

    if((originIndex.x + 0.5) * _gridSpacing <= position.x)
    {
        nearbyBucketInexes[4] += 1;
        nearbyBucketInexes[5] += 1;
        nearbyBucketInexes[6] += 1;
        nearbyBucketInexes[7] += 1;
    }
    else
    {
        nearbyBucketInexes[4] -= 1;
        nearbyBucketInexes[5] -= 1;
        nearbyBucketInexes[6] -= 1;
        nearbyBucketInexes[7] -= 1;
    }

    if((originIndex.y + 0.5) * _gridSpacing <= position.y)
    {
        nearbyBucketInexes[2] += 1;
        nearbyBucketInexes[3] += 1;
        nearbyBucketInexes[6] += 1;
        nearbyBucketInexes[7] += 1;
    }
    else
    {
        nearbyBucketInexes[2] -= 1;
        nearbyBucketInexes[3] -= 1;
        nearbyBucketInexes[6] -= 1;
        nearbyBucketInexes[7] -= 1;
    }

    if((originIndex.z + 0.5) * _gridSpacing <= position.z)
    {
        nearbyBucketInexes[1] += 1;
        nearbyBucketInexes[3] += 1;
        nearbyBucketInexes[5] += 1;
        nearbyBucketInexes[7] += 1;
    }
    else
    {
        nearbyBucketInexes[1] -= 1;
        nearbyBucketInexes[3] -= 1;
        nearbyBucketInexes[5] -= 1;
        nearbyBucketInexes[7] -= 1;
    }

    for(size_t i = 0; i < 8; i++)
    {
        nearbyKeys[i] = GetHashKeyFromBucketIndex(nearbyBucketInexes[i]);
    }
}
