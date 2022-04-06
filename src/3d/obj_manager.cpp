#include "obj_manager.hpp"


OBJManager::OBJManager()
{

}

OBJManager::~OBJManager()
{

}

void OBJManager::OnSave(std::ofstream* f, const TriangleMesh& obj)
{
    SaveHeader(f);
    SaveObjectName(f, obj);
    SaveVericies(f, obj);
    SaveNormals(f, obj);
    DisableSmoothNormals(f);
    SaveTriangles(f, obj);
}

void OBJManager::OnLoad(std::ifstream* f, TriangleMesh* obj)
{
    std::string line;
    while(std::getline(*f, line))
    {
        ParseLine(line, obj);
    }
}

void OBJManager::SaveHeader(std::ofstream* f)
{
    (*f) << "mtllib test.mtl\n";
}

void OBJManager::SaveObjectName(std::ofstream* f, const TriangleMesh& obj)
{
    (*f) << "o " << obj.GetObjectName() << "\n";
        (*f) << std::endl;
}

void OBJManager::SaveVericies(std::ofstream* f, const TriangleMesh& obj)
{
    auto& verticies = obj.GetVerticies();
    for(size_t i = 0; i < verticies.size(); i++)
    {
        (*f) << "v " << std::to_string(verticies[i].x) << " " 
                     << std::to_string(verticies[i].y) << " " 
                     << std::to_string(verticies[i].z) << "\n";
    }
    (*f) << std::endl;
}

void OBJManager::SaveNormals(std::ofstream* f, const TriangleMesh& obj)
{
    auto& normals = obj.GetNormals();
    for(size_t i = 0; i < normals.size(); i++)
    {
        (*f) << "vn " << std::to_string(normals[i].x) << " " 
                      << std::to_string(normals[i].y) << " " 
                      << std::to_string(normals[i].z) << "\n";
    }
    (*f) << std::endl;
}

void OBJManager::DisableSmoothNormals(std::ofstream* f)
{
    (*f) << "s off\n";
    (*f) << std::endl;
}

void OBJManager::SaveTriangles(std::ofstream* f, const TriangleMesh& obj)
{
    const std::vector<Triangle3D_t>& triangles = obj.GetTriangles();
    for(size_t i = 0; i < triangles.size(); i++)
    {
        SaveTriangle(f, triangles[i]);
    }
    (*f) << std::endl;
}

void OBJManager::SaveTriangle(std::ofstream* f, const Triangle3D_t& triangle)
{
    (*f) << "f ";
    (*f) << triangle.point1Idx + 1 << "//" << triangle.normalIdx + 1 << " ";
    (*f) << triangle.point2Idx + 1 << "//" << triangle.normalIdx + 1 << " ";
    (*f) << triangle.point3Idx + 1 << "//" << triangle.normalIdx + 1 << "\n";
}

void OBJManager::ParseLine(const std::string& line, TriangleMesh* obj)
{
    std::vector<std::string> splittedLine = SplitLine(line);
    if(splittedLine.size() < 1) return;
    else if(IsObjectName(splittedLine)) ParseObjectName(splittedLine, obj);
    else if(IsVertex(splittedLine)) ParseVertex(splittedLine, obj);
    else if(IsNormal(splittedLine)) ParseNormal(splittedLine, obj);
    else if(IsTriangle(splittedLine)) ParseTriangle(splittedLine, obj);
}

std::vector<std::string> OBJManager::SplitLine(const std::string& line, char delimiter)
{
    std::istringstream lineStream(line);
    std::string element;
    std::vector<std::string> strArr;
    while(std::getline(lineStream, element, delimiter)) strArr.push_back(element);
    return strArr;
}

void OBJManager::ParseObjectName(const std::vector<std::string>& line, TriangleMesh* obj)
{
    obj->SetObjectName(line[1]);
}

void OBJManager::ParseVertex(const std::vector<std::string>& line, TriangleMesh* obj)
{
    double x = std::stod(line[1]);
    double y = std::stod(line[2]);
    double z = std::stod(line[3]);
    obj->AddVertex(Vector3<double>(x, y, z));
}

void OBJManager::ParseNormal(const std::vector<std::string>& line, TriangleMesh* obj)
{
    double x = std::stod(line[1]);
    double y = std::stod(line[2]);
    double z = std::stod(line[3]);
    obj->AddNormal(Vector3<double>(x, y, z));
}

void OBJManager::ParseTriangle(const std::vector<std::string>& line, TriangleMesh* obj)
{
    std::vector<std::string> faceVertex;
    std::vector<size_t> verticies;
    for(size_t i = 1; i < line.size(); i++)
    {
        faceVertex = SplitLine(line[i], '/');
        if(faceVertex.size() > 1) verticies.push_back(std::stoul(faceVertex[0]));
    }
    obj->AddTriangle(Triangle3D_t(verticies[0] - 1, verticies[1] - 1, verticies[2] - 1, std::stoul(faceVertex[faceVertex.size() - 1]) - 1));
}

bool OBJManager::IsObjectName(const std::vector<std::string>& line)
{
    return line[0] == "o" && line.size() > 1;
}

bool OBJManager::IsVertex(const std::vector<std::string>& line)
{
    return line[0] == "v" && line.size() > 3;
}

bool OBJManager::IsNormal(const std::vector<std::string>& line)
{
    return line[0] == "vn" && line.size() > 3;
}

bool OBJManager::IsTriangle(const std::vector<std::string>& line)
{
    return line[0] == "f" && line.size() == 4;
}
