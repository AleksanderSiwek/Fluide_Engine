#include "directional_field.hpp"

DirectionalField::DirectionalField(Vector3<double> strength) : _strength(strength)
{

}

DirectionalField::DirectionalField(const DirectionalField& field) : _strength(field.Sample(0))
{

}

Vector3<double> DirectionalField::Sample(const Vector3<double>& position) const
{
    return _strength;
}

void DirectionalField::SetStrength(Vector3<double> strength)
{
    _strength = strength;
}
