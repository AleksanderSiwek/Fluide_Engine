#include "mesh_file_manager.hpp"


MeshFileManager::MeshFileManager() 
{

}

MeshFileManager::~MeshFileManager() 
{

}

void MeshFileManager::Save(std::string name, const Mesh& obj)
{
    std::ofstream f(name);
    if(CheckOpenFileError(&f))
    {
        OnSave(&f, obj);
        CheckBadBitError(&f);
    }
}

void MeshFileManager::Load(std::string name, Mesh* obj)
{
    std::ifstream f(name);
    if(CheckOpenFileError(&f))
    {
        OnLoad(&f, obj);
        CheckBadBitError(&f);
    }
}

bool MeshFileManager::CheckOpenFileError(std::ifstream* f)
{
    if(!f->is_open())
    {
        std::cout << "Error: Could not open file.\n";
        return false;
    }
    return true;
}

void MeshFileManager::CheckBadBitError(std::ifstream* f)
{
    if(f->bad())
        std::cout << "Error: Badbit occured during reading file.\n";
}

bool MeshFileManager::CheckOpenFileError(std::ofstream* f)
{
    if(!f->is_open())
    {
        std::cout << "Error: Could not open file.\n";
        return false;
    }
    return true;
}

void MeshFileManager::CheckBadBitError(std::ofstream* f)
{

    if(f->bad())
        std::cout << "Error: Badbit occured during saving file.\n";
}
