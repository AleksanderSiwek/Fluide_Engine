#include "mesh.hpp"


Mesh::Mesh()
{

}

Mesh::~Mesh()
{
    
}

void Mesh::AddVertex(const Vector3<double>& vertex)
{
    _vertices.push_back(vertex);
}

void Mesh::AddFace(const Face& face)
{
    _faces.push_back(face);
}

std::string Mesh::GetObjectName() const
{
    return _objectName;
}

const std::vector<Vector3<double>>& Mesh::GetVerticies() const
{
    return _vertices;
}

const std::vector<Vector3<double>>& Mesh::GetNormals() const
{
    return _normals;
}

const std::vector<Face>& Mesh::GetFaces() const
{
    return _faces;
}

void Mesh::SetObjectName(std::string objectName)
{
    _objectName = objectName;
}