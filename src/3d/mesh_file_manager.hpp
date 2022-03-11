#ifndef _MESH_FILE_MANAGER_HPP
#define _MESH_FILE_MANAGER_HPP

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "mesh.hpp"


class MeshFileManager
{
    public:
        MeshFileManager();
        virtual ~MeshFileManager();

        void Save(std::string name, const Mesh& obj);
        void Load(std::string name, Mesh* obj);

    protected:
        virtual void OnSave(std::ofstream* f, const Mesh& obj) = 0;
        virtual void OnLoad(std::ifstream* f, Mesh* obj) = 0;

    private:
        bool IsOpenFileError(std::ifstream* f);
        bool IsOpenFileError(std::ofstream* f);
        void CheckBadBitError(std::ifstream* f);
        void CheckBadBitError(std::ofstream* f);
};

#endif // _MESH_FILE_MANAGER_HPP