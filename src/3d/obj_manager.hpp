#ifndef _OBJ_MANAGER_HPP
#define _OBJ_MANAGER_HPP

#include "mesh_file_manager.hpp"


class OBJManager : public MeshFileManager
{
    public:
        OBJManager();

        ~OBJManager();

    protected:
        void OnSave(std::ofstream* f, const Mesh& obj) override;
        void OnLoad(std::ifstream* f, Mesh* obj) override;

    private:
        void SaveHeader(std::ofstream* f);
        void SaveObjectName(std::ofstream* f, const Mesh& obj);
        void SaveVericies(std::ofstream* f, const Mesh& obj);
        void SaveNormals(std::ofstream* f, const Mesh& obj);
        void DisableSmoothNormals(std::ofstream* f);
        void SaveFaces(std::ofstream* f, const Mesh& obj);
        void SaveFace(std::ofstream* f, const std::vector<size_t>& face, size_t normalIdx);
};

#endif // _OBJ_MANAGER_HPP