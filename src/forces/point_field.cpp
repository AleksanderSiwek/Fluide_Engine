#include "point_field.hpp"

PointField::PointField(Vector3<double> origin, double strength, double strengthFallOff) : _origin(origin), _strength(strength), _strengthFallOff(strengthFallOff)
{

}

PointField::PointField(const PointField& field) : _origin(field.GetOrigin()), _strength(field.GetStrength()), _strengthFallOff(field.GetStrengthFallOff())
{

}

PointField::~PointField() 
{

}

void PointField::SetPosition(Vector3<double> origin)
{
    _origin = origin;
}

Vector3<double> PointField::Sample(const Vector3<double>& position) const
{
    return _strength * (GetDistance(position) / _strengthFallOff) * GetDirection(position);
}

void PointField::SetStrength(double strength)
{
    _strength = strength;
}

void PointField::SetStrengthFallOff(double strengthFallOff)
{
    _strengthFallOff = strengthFallOff;
}

Vector3<double> PointField::GetOrigin() const
{
    return _origin;
}

double PointField::GetStrength() const
{
    return _strength;
}

double PointField::GetStrengthFallOff() const
{
    return _strengthFallOff;
}

Vector3<double> PointField::GetDistance(Vector3<double> point) const
{
    return (point - _origin).GetLength();
}

Vector3<double> PointField::GetDirection(Vector3<double> point) const
{
    Vector3<double> direction = (_origin - point);
    direction.Normalize();
    return direction;
}
