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

void OBJManager::SaveObjectName(std::ofstream* f, const Mesh& obj)
{
    (*f) << "o " << obj.GetObjectName() << "\n";
        (*f) << std::endl;
}

void OBJManager::SaveVericies(std::ofstream* f, const Mesh& obj)
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

void OBJManager::SaveNormals(std::ofstream* f, const Mesh& obj)
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

void OBJManager::SaveFaces(std::ofstream* f, const Mesh& obj)
{
    const std::vector<std::vector<size_t>>& faces = obj.GetFaces();
    for(size_t i = 0; i < faces.size(); i++)
    {
        SaveFace(f, faces[i], i + 1);
    }
    (*f) << std::endl;
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

void OBJManager::ParseLine(const std::string& line, Mesh* obj)
{
    std::vector<std::string> splittedLine = SplitLine(line);
    if(IsObjectName(splittedLine)) ParseObjectName(splittedLine, obj);
    else if(IsVertex(splittedLine)) ParseVertex(splittedLine, obj);
    else if(IsNormal(splittedLine)) ParseNormal(splittedLine, obj);
    else if(IsFace(splittedLine)) ParseFace(splittedLine, obj);
}

std::vector<std::string> OBJManager::SplitLine(const std::string& line, char delimiter)
{
    std::istringstream lineStream(line);
    std::string element;
    std::vector<std::string> strArr;
    while(std::getline(lineStream, element, delimiter)) strArr.push_back(element);
    return strArr;
}

void OBJManager::ParseObjectName(const std::vector<std::string>& line, Mesh* obj)
{
    obj->SetObjectName(line[1]);
}

void OBJManager::ParseVertex(const std::vector<std::string>& line, Mesh* obj)
{
    double x = std::stod(line[1]);
    double y = std::stod(line[2]);
    double z = std::stod(line[3]);
    obj->AddVertex(Vector3<double>(x, y, z));
}

void OBJManager::ParseNormal(const std::vector<std::string>& line, Mesh* obj)
{
    double x = std::stod(line[1]);
    double y = std::stod(line[2]);
    double z = std::stod(line[3]);
    obj->AddNormal(Vector3<double>(x, y, z));
}

void OBJManager::ParseFace(const std::vector<std::string>& line, Mesh* obj)
{
    std::vector<size_t> face;
    std::vector<std::string> faceVertex;
    for(size_t i = 1; i < line.size(); i++)
    {
        faceVertex = SplitLine(line[i], '/');
        if(faceVertex.size() > 1) face.push_back(std::stoul(faceVertex[0]));
    }
    obj->AddFace(face);
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

bool OBJManager::IsFace(const std::vector<std::string>& line)
{
    return line[0] == "f" && line.size() > 3;
}
