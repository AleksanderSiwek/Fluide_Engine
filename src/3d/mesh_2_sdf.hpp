#ifndef _MESH_2_SDF_HPP
#define _MESH_2_SDF_HPP

#include "triangle_mesh.hpp"
#include "scalar_grid3d.hpp"
#include "collisions.hpp"


class Mesh2SDF
{
    public:
        Mesh2SDF();

        ~Mesh2SDF();

        void Build(const TriangleMesh& mesh, ScalarGrid3D& sdf);
    
    private:
        Array3<size_t> _closestTriangles;

        void ComputeExactBandDistanceField(const TriangleMesh& mesh, ScalarGrid3D& sdf, int bandwidth);
        void ComputeSigns(const TriangleMesh& mesh, ScalarGrid3D& sdf);
};

#endif // _MESH_2_SDF_HPP