#ifndef _MESH_SIGNED_DISTANCE_FIELD_HPP
#define _MESH_SIGNED_DISTANCE_FIELD_HPP

#include "triangle_mesh.hpp"
#include "scalar_grid3d.hpp"


class MeshSignedDistanceField : public ScalarGrid3D
{
    public:
        MeshSignedDistanceField(size_t width, size_t height, size_t depth, const double& initailValue = 0, Vector3<double> origin = 0);
        MeshSignedDistanceField(Vector3<size_t> size, const double& initailValue = 0, Vector3<double> origin = 0);
        MeshSignedDistanceField(const MeshSignedDistanceField& other);

        ~MeshSignedDistanceField();

        void Build(const TriangleMesh& mesh);
    
    private:
        Array3<double> _distances;
        Array3<size_t> _closestTriangles;

        void ComputeExactBandDistanceField(const TriangleMesh& mesh, int bandwidth);
        double DistanceToTriangle(Vector3<double> pos, Vector3<double> trisPoint1, Vector3<double> trisPoint2, Vector3<double> trisPoint3);
};

#endif // _MESH_SIGNED_DISTANCE_FIELD_HPP