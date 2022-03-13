#include "triangle_mesh.hpp"


TriangleMesh::TriangleMesh()
{

}

TriangleMesh::TriangleMesh(const TriangleMesh& mesh)
{
    Set(mesh);
}

TriangleMesh::~TriangleMesh()
{

}

void TriangleMesh::Clear()
{
    _vertices.clear();
    _normals.clear();
    _triangles.clear();  
    _objectName.clear();
    _origin = 0;
}

void TriangleMesh::Set(const TriangleMesh& other)
{
    _vertices = other.GetVerticies();
    _normals = other.GetNormals();
    _triangles = other.GetTriangles();
    _objectName = other.GetObjectName();
    _origin = other.GetOrigin();
}

void TriangleMesh::AddVertex(const Vector3<double>& vertex)
{
    _vertices.push_back(vertex);
}

void TriangleMesh::AddNormal(const Vector3<double>& normal)
{
    _normals.push_back(normal);
}

void TriangleMesh::AddTriangle(const Triangle3D_t& triangle)
{
    _triangles.push_back(triangle);
}

void TriangleMesh::SetVertex(const Vector3<double>& vertex, size_t idx)
{
    if(idx >= 0 && idx < _vertices.size())
        _vertices[idx] = vertex;
}

void TriangleMesh::SetNormal(const Vector3<double>& normal, size_t idx)
{
    if(idx >= 0 && idx < _normals.size())
        _normals[idx] = normal;
}

void TriangleMesh::SetTriangle(const Triangle3D_t& triangle, size_t idx)
{
    if(idx >= 0 && idx < _triangles.size())
        _triangles[idx] = triangle;
}

std::string TriangleMesh::GetObjectName() const
{
    return _objectName;
}

Vector3<double> TriangleMesh::GetOrigin() const 
{
    return _origin;
}

const std::vector<Vector3<double>>& TriangleMesh::GetVerticies() const
{
    return _vertices;
}

const std::vector<Vector3<double>>& TriangleMesh::GetNormals() const
{
    return _normals;
}

const std::vector<Triangle3D_t>& TriangleMesh::GetTriangles() const
{
    return _triangles;
}

std::vector<Vector3<double>>& TriangleMesh::GetVerticies()
{
    return _vertices;
}

std::vector<Vector3<double>>& TriangleMesh::GetNormals()
{
    return _normals;
}

std::vector<Triangle3D_t>& TriangleMesh::GetTriangles()
{
    return _triangles;
}

void TriangleMesh::SetObjectName(std::string objectName)
{
    _objectName = objectName;
}

void TriangleMesh::SetOrigin(Vector3<double> origin)
{
    _origin = origin;
}
