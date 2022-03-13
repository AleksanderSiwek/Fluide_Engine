#ifndef _MESH_SIGNED_DISTANCE_FIELD_HPP
#define _MESH_SIGNED_DISTANCE_FIELD_HPP

#include "triangle_mesh.hpp"
#include "./scalar_grid3d.hpp"


class MeshSignedDistanceField : public ScalarGrid3D
{
    public:
        MeshSignedDistanceField(size_t width, size_t height, size_t depth, const double& initailValue = 0, Vector3<double> origin = 0);
        MeshSignedDistanceField(const Vector3<size_t>& size, const double& initailValue = 0, Vector3<double> origin = 0);
        MeshSignedDistanceField(const MeshSignedDistanceField& other);

        ~MeshSignedDistanceField();

        void Build(const TriangleMesh& mesh);
};

#endif // _MESH_SIGNED_DISTANCE_FIELD_HPP