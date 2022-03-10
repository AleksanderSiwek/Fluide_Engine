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
        ~Mesh();

        void Clear();
        void Set(const Mesh& other);
        void AddVertex(const Vector3<double>& vertex);
        void AddFace(const Face& face);

        std::string GetObjectName() const;
        const std::vector<Vector3<double>>& GetVerticies() const;
        const std::vector<Vector3<double>>& GetNormals() const;
        const std::vector<Face>& GetFaces() const;

        void SetObjectName(std::string objectName);

    private:
        std::string _objectName;
        
        std::vector<Vector3<double>> _vertices;
        std::vector<Vector3<double>> _normals;
        std::vector<Face> _faces;

        Vector3<double> origin;
};

#endif // _MESH_HPP