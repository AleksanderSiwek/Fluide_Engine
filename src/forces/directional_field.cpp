#include "directional_field.hpp"

DirectionalField::DirectionalField(Vector3<double> direction, double strength) : _direction(direction), _strength(strength)
{
    _direction.Normalize();
}

DirectionalField::DirectionalField(const DirectionalField& field) : _direction(field.GetDirection()), _strength(field.GetStrength())
{

}

void DirectionalField::SetDirection(Vector3<double> direction)
{
    _direction = direction;
}

void DirectionalField::SetStrength(double strength)
{
    _strength = strength;
}

Vector3<double> DirectionalField::GetDirection() const
{
    return _direction;
}

double DirectionalField::GetStrength() const
{
    return _strength;
}

Vector3<double> DirectionalField::RescaleVector(Vector3<double> value)
{
    return value + _direction * _strength;
}