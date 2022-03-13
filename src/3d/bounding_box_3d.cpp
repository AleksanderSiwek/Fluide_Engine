#include "bounding_box_3d.hpp"


BoundingBox3D::BoundingBox3D(Vector3<double> origin, Vector3<double> size)
    : _origin(origin), _size(size)
{

}

BoundingBox3D::~BoundingBox3D()
{

}

bool BoundingBox3D::IsInside(Vector3<double> point) const
{
    bool isInX = point.x >= _origin.x && point.x <= (_origin.x + _size.x);
    bool isInY = point.y >= _origin.y && point.y <= (_origin.y + _size.y);
    bool isInZ = point.z >= _origin.z && point.z <= (_origin.z + _size.z);
    return isInX && isInY && isInZ;
}


Vector3<double> BoundingBox3D::GetOrigin() const
{
    return _origin;
}

Vector3<double> BoundingBox3D::GetSize() const
{
    return _size;
}

void BoundingBox3D::SetOrigin(Vector3<double> origin)
{
    _origin = origin;
}

void BoundingBox3D::SetSize(Vector3<double> size)
{
    _size = size;
}