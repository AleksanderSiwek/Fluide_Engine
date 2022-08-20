#ifndef _TRIANGLE_MESH_COLLIDER_HPP
#define _TRIANGLE_MESH_COLLIDER_HPP

#include "collider.hpp"
#include "triangle_mesh.hpp"
#include "../common/array3.hpp"
#include "../3d/scalar_grid3d.hpp"
#include "../3d/mesh_2_sdf.hpp"

#include <memory>


class TriangleMeshCollider : public Collider
{
    public:
        TriangleMeshCollider(const Vector3<size_t>& gridSize, const Vector3<double>& gridOrigin, const Vector3<double>& gridSpacing, const TriangleMesh& mesh);
        ~TriangleMeshCollider();

        bool IsInside(const Vector3<double>& position) override;
        double GetClosestDistanceAt(const Vector3<double>& position) override;
        Vector3<double> GetVelocityAt(const Vector3<double>& position) override;
        void ResolveCollision(double radius, double restitutionCoefficient, const Vector3<double>& previousPosition, Vector3<double>* position, Vector3<double>* velocity) override;
        bool IsPenetraiting(const Vector3<double>& point, const Vector3<double>& closestPoint, const Vector3<double>& normal);

    private:
        Mesh2SDF _meshToSdfConverter;
        TriangleMesh _mesh;
        ScalarGrid3D _sdf;
        Array3<Vector3<double>> _velocityField;
};

#endif // _TRIANGLE_MESH_COLLIDER_HPP