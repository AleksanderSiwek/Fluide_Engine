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

Vector3<double> DirectionalField::Divergence(const Vector3<double>& position) const
{
    // TO DO
    return 0;
}

Vector3<double> DirectionalField::Curl(const Vector3<double>& position) const
{
    // TO DO
    return 0;
}

void DirectionalField::SetStrength(Vector3<double> strength)
{
    _strength = strength;
}
