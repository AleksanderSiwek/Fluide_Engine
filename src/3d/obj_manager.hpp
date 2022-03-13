#ifndef _OBJ_MANAGER_HPP
#define _OBJ_MANAGER_HPP

#include "mesh_file_manager.hpp"


class OBJManager : public MeshFileManager
{
    public:
        OBJManager();

        ~OBJManager();

    protected:
        void OnSave(std::ofstream* f, const TriangleMesh& obj) override;
        void OnLoad(std::ifstream* f, TriangleMesh* obj) override;

    private:
        void SaveHeader(std::ofstream* f);
        void SaveObjectName(std::ofstream* f, const TriangleMesh& obj);
        void SaveVericies(std::ofstream* f, const TriangleMesh& obj);
        void SaveNormals(std::ofstream* f, const TriangleMesh& obj);
        void DisableSmoothNormals(std::ofstream* f);
        void SaveTriangles(std::ofstream* f, const TriangleMesh& obj);
        void SaveTriangle(std::ofstream* f, const Triangle3D_t& triangle);

        void ParseLine(const std::string& line, TriangleMesh* obj);
        std::vector<std::string> SplitLine(const std::string& line, char delimiter = ' ');
        void ParseObjectName(const std::vector<std::string>& line, TriangleMesh* obj);
        void ParseVertex(const std::vector<std::string>& line, TriangleMesh* obj);
        void ParseNormal(const std::vector<std::string>& line, TriangleMesh* obj);
        void ParseTriangle(const std::vector<std::string>& line, TriangleMesh* obj);
        bool IsObjectName(const std::vector<std::string>& line);
        bool IsVertex(const std::vector<std::string>& line);
        bool IsNormal(const std::vector<std::string>& line);
        bool IsTriangle(const std::vector<std::string>& line);
};

#endif // _OBJ_MANAGER_HPP