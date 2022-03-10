#include "obj_manager.hpp"


OBJManager::OBJManager()
{

}

OBJManager::~OBJManager()
{

}

void OBJManager::OnSave(std::ofstream* f, const Mesh& obj)
{
    SaveHeader(f);
    SaveObjectName(f, obj);
    SaveVericies(f, obj);
    SaveNormals(f, obj);
    DisableSmoothNormals(f);
    SaveFaces(f, obj);
}

void OBJManager::OnLoad(std::ifstream* f, Mesh* obj)
{

}

void OBJManager::SaveHeader(std::ofstream* f)
{
    (*f) << "mtllib test.mtl\n";
}

void OBJManager::SaveObjectName(std::ofstream* f, const Mesh& obj)
{
    (*f) << "o " << obj.GetObjectName() << "\n";
}

void OBJManager::SaveVericies(std::ofstream* f, const Mesh& obj)
{
    auto& verticies = obj.GetVerticies();
    for(size_t i = 0; i < verticies.size(); i++)
    {
        (*f) << "v " << verticies[i].x << " " << verticies[i].y << " " << verticies[i].z << "\n";
    }
}

void OBJManager::SaveNormals(std::ofstream* f, const Mesh& obj)
{
    auto& normals = obj.GetNormals();
    for(size_t i = 0; i < normals.size(); i++)
    {
        (*f) << "vn " << normals[i].x << " " << normals[i].y << " " << normals[i].z << "\n";
    }
}

void OBJManager::DisableSmoothNormals(std::ofstream* f)
{
    (*f) << "s off\n";
}

void OBJManager::SaveFaces(std::ofstream* f, const Mesh& obj)
{
    const std::vector<std::vector<size_t>>& faces = obj.GetFaces();
    for(size_t i = 0; i < faces.size(); i++)
    {
        SaveFace(f, faces[i], i);
    }
}

void OBJManager::SaveFace(std::ofstream* f, const std::vector<size_t>& face, size_t normalIdx)
{
    (*f) << "f ";
    for(size_t j = 0; j < face.size(); j++)
    {
        (*f) << face[j] << "//" << normalIdx << " ";
    }
    (*f) << "\n";
}

