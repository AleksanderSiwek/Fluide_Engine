#include "mesh_signed_distance_field.hpp"


MeshSignedDistanceField::MeshSignedDistanceField(size_t width, size_t height, size_t depth, const double& initailValue, Vector3<double> origin)
    : ScalarGrid3D(width, height, depth, initailValue, origin), _closestTriangles(width+1, height+1, depth+1), _distances(width+1, height+1, depth+1)
{

}

MeshSignedDistanceField::MeshSignedDistanceField(Vector3<size_t> size, const double& initailValue, Vector3<double> origin)
    : ScalarGrid3D(size, initailValue, origin), _closestTriangles(size+size_t(1)), _distances(size+size_t(1))
{

}

MeshSignedDistanceField::MeshSignedDistanceField(const MeshSignedDistanceField& other)
    : ScalarGrid3D(other)
{

}

MeshSignedDistanceField::~MeshSignedDistanceField()
{

}

void MeshSignedDistanceField::Build(const TriangleMesh& mesh)
{

}

void MeshSignedDistanceField::ComputeExactBandDistanceField(const TriangleMesh& mesh, int bandwidth)
{

    _distances.Fill(_gridSpacing.Max() * (_size.x + _size.y + _size.z));
    _closestTriangles.Fill(-1);
    Vector3<size_t> size = _distances.GetSize();

    const auto& verticies = mesh.GetVerticies();
    const auto& triangles = mesh.GetTriangles();
    if(verticies.empty()) 
        return;

    Triangle3D_t tris;
    double inversedGridSpacing = 1.0 / _gridSpacing.x;
    for(size_t trisIdx = 0; trisIdx < triangles.size(); trisIdx++)
    {
        tris = triangles[trisIdx];
        Vector3<double> vertexPos1 = verticies[tris.point1Idx] * inversedGridSpacing;
        Vector3<double> vertexPos2 = verticies[tris.point2Idx] * inversedGridSpacing;
        Vector3<double> vertexPos3 = verticies[tris.point3Idx] * inversedGridSpacing;

        int i0 = Clamp<int>(int(fmin(vertexPos1.x, fmin(vertexPos2.x, vertexPos3.x))) - bandwidth, 0, (int)size.x - 1);
        int j0 = Clamp<int>(int(fmin(vertexPos1.y, fmin(vertexPos2.y, vertexPos3.y))) - bandwidth, 0, (int)size.y - 1);
        int k0 = Clamp<int>(int(fmin(vertexPos1.z, fmin(vertexPos2.z, vertexPos3.z))) - bandwidth, 0, (int)size.z - 1);

        int iMax = Clamp<int>(int(fmax(vertexPos1.x, fmax(vertexPos2.x, vertexPos3.x))) + bandwidth, 0, (int)size.x - 1);
        int jMax = Clamp<int>(int(fmax(vertexPos1.y, fmax(vertexPos2.y, vertexPos3.y))) + bandwidth, 0, (int)size.y - 1);
        int kMax = Clamp<int>(int(fmax(vertexPos1.z, fmax(vertexPos2.z, vertexPos3.z))) + bandwidth, 0, (int)size.z - 1);

        for(int i = i0; i <= iMax; i++)
        {
            for(int j = j0; j <= jMax; j++)
            {
                for(int k = k0; k <= kMax; k++)
                {
                    double distance = DistanceToTriangle(GridIndexToPosition(i, j, k), verticies[tris.point1Idx], verticies[tris.point2Idx], verticies[tris.point3Idx]);
                    if(distance < _distances(i, j ,k))
                    {
                        if(fabs(distance) < fabs(_distances(i, j, k)))
                            _closestTriangles(i, j, k) = trisIdx;
                        _distances(i, j, k) = distance;
                    }
                }
            }
        }

    }
}

double MeshSignedDistanceField::DistanceToTriangle(Vector3<double> pos, Vector3<double> trisPoint1, Vector3<double> trisPoint2, Vector3<double> trisPoint3)
{
    return 0;
}

