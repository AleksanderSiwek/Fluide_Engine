#ifndef _MESH_SIGNED_DISTANCE_FIELD_HPP
#define _MESH_SIGNED_DISTANCE_FIELD_HPP

#include "mesh.hpp"
#include "./scalar_grid3.hpp"


class MeshSignedDistanceField : public ScalarGrid3
{
    public:
        MeshSignedDistanceField(size_t width, size_t height, size_t depth, const double& initailValue = 0, Vector3<double> origin = 0);
        MeshSignedDistanceField(const Vector3<size_t>& size, const double& initailValue = 0, Vector3<double> origin = 0);
        MeshSignedDistanceField(const MeshSignedDistanceField& other);

        ~MeshSignedDistanceField();

        void Build(const Mesh& mesh);
};

#endif // _MESH_SIGNED_DISTANCE_FIELD_HPP