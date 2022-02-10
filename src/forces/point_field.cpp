#include "point_field.hpp"

PointField::PointField(Vector3<double> position, double strength, double strengthFallOff) : _position(position), _strength(strength), _strengthFallOff(strengthFallOff)
{

}

PointField::PointField(const PointField& field) : _position(field.GetPosition()), _strength(field.GetStrength()), _strengthFallOff(field.GetStrengthFallOff())
{

}

PointField::~PointField() 
{

}

void PointField::SetPosition(Vector3<double> position)
{
    _position = position;
}

void PointField::SetStrength(double strength)
{
    _strength = strength;
}

void PointField::SetStrengthFallOff(double strengthFallOff)
{
    _strengthFallOff = strengthFallOff;
}

Vector3<double> PointField::GetPosition() const
{
    return _position;
}

double PointField::GetStrength() const
{
    return _strength;
}

double PointField::GetStrengthFallOff() const
{
    return _strengthFallOff;
}

Vector3<double> PointField::RescaleVector(Vector3<double> point, Vector3<double> value) const
{
    return value + _strength * (GetDistance(point) / _strengthFallOff) * GetDirection(point);
}

Vector3<double> PointField::GetDistance(Vector3<double> point) const
{
    return (point - _position).GetLength();
}

Vector3<double> PointField::GetDirection(Vector3<double> point) const
{
    Vector3<double> direction = (_position - point);
    direction.Normalize();
    return direction;
}