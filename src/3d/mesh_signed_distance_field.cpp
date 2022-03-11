#include "mesh_signed_distance_field.hpp"


MeshSignedDistanceField::MeshSignedDistanceField(size_t width, size_t height, size_t depth, const double& initailValue, Vector3<double> origin)
    : ScalarGrid3(width, height, depth, initailValue, origin)
{

}

MeshSignedDistanceField::MeshSignedDistanceField(const Vector3<size_t>& size, const double& initailValue, Vector3<double> origin)
    : ScalarGrid3(size, initailValue, origin)
{

}

MeshSignedDistanceField::MeshSignedDistanceField(const MeshSignedDistanceField& other)
    : ScalarGrid3(other)
{

}

MeshSignedDistanceField::~MeshSignedDistanceField()
{

}

void MeshSignedDistanceField::Build(const Mesh& mesh)
{

}
