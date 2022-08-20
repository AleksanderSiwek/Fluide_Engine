#include "triangle_mesh_collider.hpp"
#include "mesh_2_sdf.hpp"


TriangleMeshCollider::TriangleMeshCollider(const Vector3<size_t>& gridSize, const Vector3<double>& gridOrigin, const Vector3<double>& gridSpacing, const TriangleMesh& mesh)
    : _mesh(mesh), _sdf(gridSize, 0, gridOrigin, gridSpacing), _velocityField(gridSize, 0)
{
    _meshToSdfConverter.Build(mesh, _sdf);
}

TriangleMeshCollider::~TriangleMeshCollider()
{

}

bool TriangleMeshCollider::IsInside(const Vector3<double>& position)
{
    return _sdf.Sample(position) < 0;
}

double TriangleMeshCollider::GetClosestDistanceAt(const Vector3<double>& position)
{
    return _sdf.Sample(position);
}

Vector3<double> TriangleMeshCollider::GetVelocityAt(const Vector3<double>& position)
{
    // const auto& origin = _sdf.GetOrigin();
    // const auto& gridSpacing = _sdf.GetGridSpacing();
    // int i, j, k;
    // double factorX, factorY, factorZ ;
    // i = j = k = 0;
    // factorX = factorY = factorZ = 0;

    // Vector3<double> normalizedPoistion = (position - origin) / gridSpacing;
    // const auto& size = _sdf.GetSize();
    // int sizeX = static_cast<int>(size.x);
    // int sizeY = static_cast<int>(size.y);
    // int sizeZ = static_cast<int>(size.z);

    // GetBarycentric<double>(normalizedPoistion.x, 0, sizeX - 1, &i, &factorX);
    // GetBarycentric<double>(normalizedPoistion.y, 0, sizeY - 1, &j, &factorY);
    // GetBarycentric<double>(normalizedPoistion.z, 0, sizeZ - 1, &k, &factorZ);

    // size_t ip1 = std::min(i + 1, sizeX - 1);
    // size_t jp1 = std::min(j + 1, sizeY - 1);
    // size_t kp1 = std::min(k + 1, sizeZ - 1);

    // return Trilerp<Vector3<double>, double>( 
    //                                 _velocityField(i, j, k),
    //                                 _velocityField(ip1, j, k),
    //                                 _velocityField(i, jp1, k),
    //                                 _velocityField(ip1, jp1, k),
    //                                 _velocityField(i, j, kp1),
    //                                 _velocityField(ip1, j, kp1),
    //                                 _velocityField(i, jp1, kp1),
    //                                 _velocityField(ip1, jp1, kp1),
    //                                 factorX,
    //                                 factorY,
    //                                 factorZ);
    return 0;
}

#include <iostream>
void TriangleMeshCollider::ResolveCollision(double radius, double restitutionCoefficient, const Vector3<double>& previousPosition, Vector3<double>* position, Vector3<double>* velocity)
{
    double frictionCoeffient = 0; // TO DO
    Vector3<double> colliderVelocity(0.0, 0.0, 0.0); // TO DO

    const auto& triangles = _mesh.GetTriangles();
    const auto& verts = _mesh.GetVerticies();
    const auto& normals = _mesh.GetNormals();
    size_t triangleIdx = Collisions::ClosestTriangleIdx(previousPosition, _mesh);
    Vector3<double> closestPoint = Collisions::ClossestPointOnTriangle(previousPosition, verts[triangles[triangleIdx].point1Idx], verts[triangles[triangleIdx].point2Idx], verts[triangles[triangleIdx].point3Idx]);
    Vector3<double> closestNormal = normals[triangles[triangleIdx].normalIdx];

    if(_mesh.IsInside(*position))
    {
        Vector3<double> relativeVel = *velocity - colliderVelocity;
        double normalDotRelativeVel = closestNormal.Dot(relativeVel);
        Vector3<double> relativeVelN = normalDotRelativeVel * closestNormal;
        Vector3<double> relativeVelT = relativeVel - relativeVelN;

        if(normalDotRelativeVel < 0.0)
        {
            Vector3<double> deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN;
            relativeVelN *= -restitutionCoefficient;

            if(relativeVelT.GetLengthSquared() > 0.0)
            {
                double frictionScale =  std::max(
                    1.0 - frictionCoeffient * deltaRelativeVelN.GetLength() / relativeVelT.GetLength(),
                    0.0);
                    relativeVelT *= frictionScale;
            }
        }
            *velocity = relativeVelN + relativeVelT + colliderVelocity;
    }
    *position = closestPoint;
}

bool TriangleMeshCollider::IsPenetraiting(const Vector3<double>& point, const Vector3<double>& closestPoint, const Vector3<double>& normal)
{
    return (point - closestPoint).Dot(normal) < 0.0;
}

