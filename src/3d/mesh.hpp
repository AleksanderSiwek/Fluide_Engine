#ifndef _MESH_HPP
#define _MESH_HPP

#include <vector>
#include <string>

#include "../common/vector3.hpp"

typedef std::vector<size_t> Face;


class Mesh
{
    public:
        Mesh();
        Mesh(const Mesh& mesh);
        ~Mesh();

        void Clear();
        void Set(const Mesh& other);
        void AddVertex(const Vector3<double>& vertex);
        void AddNormal(const Vector3<double>& normal);
        void AddFace(const Face& face);
        void SetVertex(const Vector3<double>& vertex, size_t idx);
        void SetNormal(const Vector3<double>& normal, size_t idx);
        void SetFace(const Face& face, size_t idx);

        std::string GetObjectName() const;
        const std::vector<Vector3<double>>& GetVerticies() const;
        const std::vector<Vector3<double>>& GetNormals() const;
        const std::vector<Face>& GetFaces() const;
        std::vector<Vector3<double>>& GetVerticies();
        std::vector<Vector3<double>>& GetNormals();
        std::vector<Face>& GetFaces();

        void SetObjectName(std::string objectName);

    private:
        std::string _objectName;
        
        std::vector<Vector3<double>> _vertices;
        std::vector<Vector3<double>> _normals;
        std::vector<Face> _faces;

        Vector3<double> origin;
};

#endif // _MESH_HPP