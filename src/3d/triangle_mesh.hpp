#ifndef _TRIANGLE_MESH_HPP
#define _TRIANGLE_MESH_HPP

#include <vector>
#include <string>

#include "triangle_3d.hpp"
#include "../common/vector3.hpp"


class TriangleMesh
{
    public:
        TriangleMesh();
        TriangleMesh(const TriangleMesh& mesh);

        ~TriangleMesh();

        void Clear();
        void Set(const TriangleMesh& other);
        void AddVertex(const Vector3<double>& vertex);
        void AddNormal(const Vector3<double>& normal);
        void AddTriangle(const Triangle3D_t& triangle);
        void SetVertex(const Vector3<double>& vertex, size_t idx);
        void SetNormal(const Vector3<double>& normal, size_t idx);
        void SetTriangle(const Triangle3D_t& triangle, size_t idx);

        std::string GetObjectName() const;
        Vector3<double> GetOrigin() const;
        const std::vector<Vector3<double>>& GetVerticies() const;
        const std::vector<Vector3<double>>& GetNormals() const;
        const std::vector<Triangle3D_t>& GetTriangles() const;
        std::vector<Vector3<double>>& GetVerticies();
        std::vector<Vector3<double>>& GetNormals();
        std::vector<Triangle3D_t>& GetTriangles();

        void SetObjectName(std::string objectName);
        void SetOrigin(Vector3<double> origin);

    private:
        std::string _objectName;
        
        std::vector<Vector3<double>> _vertices;
        std::vector<Vector3<double>> _normals;
        std::vector<Triangle3D_t> _triangles;

        Vector3<double> _origin;
};

#endif // _TRIANGLE_MESH_HPP