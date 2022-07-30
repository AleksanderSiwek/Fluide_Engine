#include "collider_collection.hpp"

#define MAX_DISTANCE 1.7976931348623157E+30

ColliderCollection::ColliderCollection(const Vector3<size_t>& size, const Vector3<double>& gridSpacing, const Vector3<double>& origin) 
    : _size(size), _gridSpacing(gridSpacing), _origin(origin), _sdf(size, 0, origin, gridSpacing)
{

}

ColliderCollection::~ColliderCollection()
{

}

bool ColliderCollection::IsInside(const Vector3<double>& position)
{
    for(size_t i = 0; i < _colliders.size(); i++)
    {
        if(_colliders[i]->IsInside(position))
        {
            return true;
        }
    }
    return false;
}

void ColliderCollection::AddCollider(std::shared_ptr<Collider> collider)
{
    _colliders.push_back(collider);
}

void ColliderCollection::BuildSdf()
{
    _sdf.Resize(_size);
    _sdf.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        double minDistance = MAX_DISTANCE;
        const auto& position = _sdf.GridIndexToPosition(i, j, k);
        for(size_t colliderIdx = 0; colliderIdx < _colliders.size(); colliderIdx++)
        {
            double distance = _colliders[colliderIdx]->GetClosestDistanceAt(position);
            if(distance < minDistance)
            {
                minDistance = distance;
            }
        }
        _sdf(i, j, k) = minDistance;
    });
}

const ScalarGrid3D& ColliderCollection::GetSdf() const
{
    return _sdf;
}

std::vector<std::shared_ptr<Collider>>& ColliderCollection::GetColliders()
{
    return _colliders;
}

