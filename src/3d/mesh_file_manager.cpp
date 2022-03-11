#include "mesh_file_manager.hpp"


MeshFileManager::MeshFileManager() 
{

}

MeshFileManager::~MeshFileManager() 
{

}

void MeshFileManager::Save(std::string name, const Mesh& obj)
{
    std::ofstream f(name.c_str());
    if(IsOpenFileError(&f))
    {
        OnSave(&f, obj);
        CheckBadBitError(&f);
    }
}

void MeshFileManager::Load(std::string name, Mesh* obj)
{
    std::ifstream f(name.c_str());
    if(IsOpenFileError(&f))
    {
        OnLoad(&f, obj);
        CheckBadBitError(&f);
    }
}

bool MeshFileManager::IsOpenFileError(std::ifstream* f)
{
    if(f->is_open() && f->good())
        return true;
    std::cout << "Error Load(): Could not open file.\n";
    return false;
}

void MeshFileManager::CheckBadBitError(std::ifstream* f)
{
    if(f->bad())
        std::cout << "Error Load(): Badbit occured during reading file.\n";
}

bool MeshFileManager::IsOpenFileError(std::ofstream* f)
{
    if(f->is_open() && f->good())
        return true;
    std::cout << "Error Save(): Could not open file.\n";
    return false;
}

void MeshFileManager::CheckBadBitError(std::ofstream* f)
{

    if(f->bad())
        std::cout << "Error Save(): Badbit occured during saving file.\n";
}
