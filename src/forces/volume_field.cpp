#include "volume_field.hpp"

#include "../3d/mesh_2_sdf.hpp"
#include "../3d/collisions.hpp"

#include <iostream>


VolumeField::VolumeField(const TriangleMesh& mesh, Vector3<size_t> resolution, BoundingBox3D domain, double strength)
    : _mesh(mesh), _strength(strength), _resolution(resolution)
{
    auto domainSize = domain.GetSize();
    _gridSpacing = Vector3<double>(domainSize.x / resolution.x, domainSize.y / resolution.y, domainSize.z / resolution.z);
    _origin = domain.GetOrigin();
    _sdf = ScalarGrid3D(_resolution, 0, _origin, _gridSpacing);
    Initialize();
}


VolumeField::~VolumeField()
{

}

Vector3<double> VolumeField::Sample(const Vector3<double>& position) const
{
    // std::cout << "Sample!\n";
    int i, j, k;
    double factorX, factorY, factorZ;
    i = j = k = 0;
    factorX = factorY = factorZ = 0;

    Vector3<double> normalizedPoistion = (position - _origin) / _gridSpacing;
    const auto& size = _resolution;
    int sizeX = static_cast<int>(size.x);
    int sizeY = static_cast<int>(size.y);
    int sizeZ = static_cast<int>(size.z);

    GetBarycentric<double>(normalizedPoistion.x, 0, sizeX - 1, &i, &factorX);
    GetBarycentric<double>(normalizedPoistion.y, 0, sizeY - 1, &j, &factorY);
    GetBarycentric<double>(normalizedPoistion.z, 0, sizeZ - 1, &k, &factorZ);

    size_t ip1 = std::min(i + 1, sizeX - 1);
    size_t jp1 = std::min(j + 1, sizeY - 1);
    size_t kp1 = std::min(k + 1, sizeZ - 1);

    // std::cout << "i: " << _resolution.x << ", j: " << _resolution.y << ", k: " << _resolution.z << "\n";

    Vector3<double> sampledForce = Trilerp<Vector3<double>, double>(_vectorField(i, j, k),
                                                                    _vectorField(ip1, j, k),
                                                                    _vectorField(i, jp1, k),
                                                                    _vectorField(ip1, jp1, k),
                                                                    _vectorField(i, j, kp1),
                                                                    _vectorField(ip1, j, kp1),
                                                                    _vectorField(i, jp1, kp1),
                                                                    _vectorField(ip1, jp1, kp1),
                                                                    factorX,
                                                                    factorY,
                                                                    factorZ);
                                                                    
    Vector3<double> forceDirection = sampledForce.GetNormalized() * (-1.0);
    double distanceToPoint = sampledForce.GetLength() >= 1.0 ? sampledForce.GetLength() : 1.0;
    return _strength * (1 / (distanceToPoint * distanceToPoint)) * forceDirection;
    // return 0;
    // std::cout << "Sample end!\n";
}

void VolumeField::SetStrength(double strength)
{
    _strength = strength;
}

double VolumeField::GetStrength() const
{
    return _strength;
}

void VolumeField::Initialize()
{
    std::cout << "Inicialization start\n";
    std::cout << "Resolution: " << _resolution.x << ", " << _resolution.y << ", " << _resolution.z << "\n";
    _vectorField.Resize(_resolution);
    _sdf.Resize(_resolution);

    auto converter = Mesh2SDF();
    converter.Build(_mesh, _sdf);

    const auto& triangles = _mesh.GetTriangles();
    const auto& verticies = _mesh.GetVerticies();

    _vectorField.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        size_t idx = Collisions::ClosestTriangleIdx(gp, _mesh);
        Vector3<double> cp = Collisions::ClossestPointOnTriangle(gp, verticies[triangles[idx].point1Idx], verticies[triangles[idx].point2Idx], verticies[triangles[idx].point3Idx]);
        _vectorField(i, j, k) = (cp - gp);
    });

    std::cout << "Inicialization end\n";
}
