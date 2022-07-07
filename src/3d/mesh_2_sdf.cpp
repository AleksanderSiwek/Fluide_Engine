#include "mesh_2_sdf.hpp"

Mesh2SDF::Mesh2SDF()
{

}


Mesh2SDF::~Mesh2SDF()
{

}

void Mesh2SDF::Build(const TriangleMesh& mesh, ScalarGrid3D& sdf)
{
    ComputeExactBandDistanceField(mesh, sdf, 1);
    ComputeSigns(mesh, sdf);
}

void Mesh2SDF::ComputeExactBandDistanceField(const TriangleMesh& mesh, ScalarGrid3D& sdf, int bandwidth)
{
    // TO DO: Clamping
    Vector3<size_t> size = sdf.GetSize();
    Vector3<double> gridSpacing = sdf.GetGridSpacing();
    sdf.ParallelFill(sdf.GetGridSpacing().Max() * (size.x + size.y + size.z));

    const auto& verticies = mesh.GetVerticies();
    const auto& triangles = mesh.GetTriangles();
    if(verticies.empty()) 
        return;

    Triangle3D_t tris;
    double inversedGridSpacing = 1.0 / gridSpacing.x;
    for(size_t trisIdx = 0; trisIdx < triangles.size(); trisIdx++)
    {
        tris = triangles[trisIdx];
        Vector3<double> vertexPos1 = verticies[tris.point1Idx] * inversedGridSpacing;
        Vector3<double> vertexPos2 = verticies[tris.point2Idx] * inversedGridSpacing;
        Vector3<double> vertexPos3 = verticies[tris.point3Idx] * inversedGridSpacing;

        // int i0 = Clamp<int>(int(fmin(vertexPos1.x, fmin(vertexPos2.x, vertexPos3.x))) - bandwidth, 0, (int)size.x - 1);
        // int j0 = Clamp<int>(int(fmin(vertexPos1.y, fmin(vertexPos2.y, vertexPos3.y))) - bandwidth, 0, (int)size.y - 1);
        // int k0 = Clamp<int>(int(fmin(vertexPos1.z, fmin(vertexPos2.z, vertexPos3.z))) - bandwidth, 0, (int)size.z - 1);

        // int iMax = Clamp<int>(int(fmax(vertexPos1.x, fmax(vertexPos2.x, vertexPos3.x))) + bandwidth, 0, (int)size.x - 1);
        // int jMax = Clamp<int>(int(fmax(vertexPos1.y, fmax(vertexPos2.y, vertexPos3.y))) + bandwidth, 0, (int)size.y - 1);
        // int kMax = Clamp<int>(int(fmax(vertexPos1.z, fmax(vertexPos2.z, vertexPos3.z))) + bandwidth, 0, (int)size.z - 1);

        parallel_utils::ForEach3(size.x, size.y, size.z, [&](size_t i, size_t j, size_t k)
        {
            double distance = Collisions::DistanceToTriangle(sdf.GridIndexToPosition(i, j, k), 
                                                                verticies[tris.point1Idx], 
                                                                verticies[tris.point2Idx], 
                                                                verticies[tris.point3Idx]);
            if(distance < sdf(i, j ,k))
                sdf(i, j, k) = distance;
        });
    }
}

void Mesh2SDF::ComputeSigns(const TriangleMesh& mesh, ScalarGrid3D& sdf)
{
    Vector3<size_t> size = sdf.GetSize(); 
    Array3<bool> isInside(size.x, size.y, size.z, false);

    parallel_utils::ForEach3(size.x, size.y, size.z, [&](size_t i, size_t j, size_t k)
    {
                if(mesh.IsInside(sdf.GridIndexToPosition(i, j, k)))
                    sdf(i, j, k) = sdf(i, j, k) * -1.0;
    });
}