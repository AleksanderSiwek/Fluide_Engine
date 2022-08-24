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

        Vector3<double> p = verticies[tris.point1Idx];
        Vector3<double> q = verticies[tris.point2Idx];
        Vector3<double> r = verticies[tris.point3Idx];

        double fip = (double)p.x * inversedGridSpacing;
        double fjp = (double)p.y * inversedGridSpacing; 
        double fkp = (double)p.z * inversedGridSpacing;

        double fiq = (double)q.x * inversedGridSpacing;
        double fjq = (double)q.y * inversedGridSpacing;
        double fkq = (double)q.z * inversedGridSpacing;

        double fir = (double)r.x * inversedGridSpacing;
        double fjr = (double)r.y * inversedGridSpacing;
        double fkr = (double)r.z * inversedGridSpacing;

        int i0 = Clamp<int>(int(fmin(fip, fmin(fiq, fir))) - bandwidth, (int)size.x - 1, 0);
        int j0 = Clamp<int>(int(fmin(fjp, fmin(fjq, fjr))) - bandwidth, (int)size.y - 1, 0);
        int k0 = Clamp<int>(int(fmin(fkp, fmin(fkq, fkr))) - bandwidth, (int)size.z - 1, 0);

        int i1 = Clamp<int>(int(fmax(fip, fmax(fiq, fir))) + bandwidth + 1, (int)size.x - 1, 0);
        int j1 = Clamp<int>(int(fmax(fjp, fmax(fjq, fjr))) + bandwidth + 1, (int)size.y - 1, 0);
        int k1 = Clamp<int>(int(fmax(fkp, fmax(fkq, fkr))) + bandwidth + 1, (int)size.z - 1, 0);

        // int i0 = Clamp<int>(int(fmin(vertexPos1.x, fmin(vertexPos2.x, vertexPos3.x))) - bandwidth, 0, (int)size.x - 1);
        // int j0 = Clamp<int>(int(fmin(vertexPos1.y, fmin(vertexPos2.y, vertexPos3.y))) - bandwidth, 0, (int)size.y - 1);
        // int k0 = Clamp<int>(int(fmin(vertexPos1.z, fmin(vertexPos2.z, vertexPos3.z))) - bandwidth, 0, (int)size.z - 1);

        // int iMax = Clamp<int>(int(fmax(vertexPos1.x, fmax(vertexPos2.x, vertexPos3.x))) + bandwidth, 0, (int)size.x - 1);
        // int jMax = Clamp<int>(int(fmax(vertexPos1.y, fmax(vertexPos2.y, vertexPos3.y))) + bandwidth, 0, (int)size.y - 1);
        // int kMax = Clamp<int>(int(fmax(vertexPos1.z, fmax(vertexPos2.z, vertexPos3.z))) + bandwidth, 0, (int)size.z - 1);

        parallel_utils::ForEach3(size.x, size.y, size.z, [&](size_t i, size_t j, size_t k)
        {
            if(i >= i0 && j >= j0 && k >= k0 && i <= i1 && j <= j1 && k <= k1)
            {
                double distance = Collisions::DistanceToTriangle(sdf.GridIndexToPosition(i, j, k), 
                                                                    verticies[tris.point1Idx], 
                                                                    verticies[tris.point2Idx], 
                                                                    verticies[tris.point3Idx]);
                if(distance < sdf(i, j ,k))
                    sdf(i, j, k) = distance;
            }
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