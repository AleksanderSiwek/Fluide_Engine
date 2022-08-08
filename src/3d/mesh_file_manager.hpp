#ifndef _MESH_FILE_MANAGER_HPP
#define _MESH_FILE_MANAGER_HPP

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "triangle_mesh.hpp"


class MeshFileManager
{
    public:
        MeshFileManager();
        virtual ~MeshFileManager();

        void Save(std::string name, const TriangleMesh& obj);
        void Load(std::string name, TriangleMesh& obj);

    protected:
        virtual void OnSave(std::ofstream* f, const TriangleMesh& obj) = 0;
        virtual void OnLoad(std::ifstream* f, TriangleMesh* obj) = 0;

    private:
        bool IsOpenFileError(std::ifstream* f);
        bool IsOpenFileError(std::ofstream* f);
        void CheckBadBitError(std::ifstream* f);
        void CheckBadBitError(std::ofstream* f);
};

#endif // _MESH_FILE_MANAGER_HPP