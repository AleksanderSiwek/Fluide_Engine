#ifndef _COLLIDER_COLLECTION_HPP
#define _COLLIDER_COLLECTION_HPP

#include <vector>
#include <memory>

#include "collider.hpp"
#include "scalar_grid3d.hpp"


class ColliderCollection
{
    public:
        ColliderCollection(const Vector3<size_t>& size, const Vector3<double>& gridSpacing, const Vector3<double>& origin);
        ~ColliderCollection();

        bool IsInside(const Vector3<double>& position);
        void AddCollider(std::shared_ptr<Collider> collider);
        void BuildSdf();

        const ScalarGrid3D& GetSdf() const;
        std::vector<std::shared_ptr<Collider>>& GetColliders();


    private:
        std::vector<std::shared_ptr<Collider>> _colliders;
        ScalarGrid3D _sdf;
        Vector3<size_t> _size;
        Vector3<double> _gridSpacing;
        Vector3<double> _origin;
};



#endif // _COLLIDER_COLLECTION_HPP