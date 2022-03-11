#include "mesh.hpp"


Mesh::Mesh()
{

}

Mesh::Mesh(const Mesh& mesh)
{
    Set(mesh);
}

Mesh::~Mesh()
{
    
}

void Mesh::Clear()
{
    _vertices.clear();
    _normals.clear();
    _faces.clear();  
}

void Mesh::Set(const Mesh& other)
{
    _vertices = other.GetVerticies();
    _normals = other.GetNormals();
    _faces = other.GetFaces();
}

void Mesh::AddVertex(const Vector3<double>& vertex)
{
    _vertices.push_back(vertex);
}

void Mesh::AddNormal(const Vector3<double>& normal)
{
    _normals.push_back(normal);
}

void Mesh::AddFace(const Face& face)
{
    _faces.push_back(face);
}

void Mesh::SetVertex(const Vector3<double>& vertex, size_t idx)
{
    if(idx < _vertices.size())
        _vertices[idx] = vertex;  
}

void Mesh::SetNormal(const Vector3<double>& normal, size_t idx)
{
    if(idx < _normals.size())
        _normals[idx] = normal;  
}

void Mesh::SetFace(const Face& face, size_t idx)
{
    if(idx < _faces.size())
        _faces[idx] = face;
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

std::vector<Vector3<double>>& Mesh::GetVerticies()
{
    return _vertices;
}

std::vector<Vector3<double>>& Mesh::GetNormals()
{
    return _normals;
}

std::vector<Face>& Mesh::GetFaces()
{
    return _faces;
}

void Mesh::SetObjectName(std::string objectName)
{
    _objectName = objectName;
}