#include "swirl_field.hpp"
#include "../3d/collisions.hpp"


SwirlField::SwirlField(Vector3<double> origin, Vector3<double> axisDirection, double strength, double strengthFallOff)
    : _origin(origin), _axisDirection(axisDirection.GetNormalized()), _strength(strength), _strengthFallOff(strengthFallOff)
{

}

SwirlField::SwirlField(const SwirlField& field)
    : _origin(field.GetOrigin()), _axisDirection(field.GetAxisDirection()), _strength(field.GetStrength()), _strengthFallOff(field.GetStrengthFallOff())
{

}

SwirlField::~SwirlField()
{

}

Vector3<double> SwirlField::Sample(const Vector3<double>& position) const
{
    double distanceToPoint = Collisions::DistanceToAxis(_origin, _axisDirection, position);
    distanceToPoint = distanceToPoint >= 1.0 ? distanceToPoint : 1.0;
    Vector3<double> rotatedPosition = Collisions::RotatePointAroundAxis(_origin, _axisDirection, position, 0.0001);
    Vector3<double> forceDirection = (position - rotatedPosition).GetNormalized();
    return _strength * (1 / (distanceToPoint * distanceToPoint)) * forceDirection;
}

void SwirlField::SetOrigin(Vector3<double> origin)
{
    _origin = origin;
}

void SwirlField::SetAxisDirection(Vector3<double> axisDirection)
{
    axisDirection = axisDirection;
}

void SwirlField::SetStrength(double strength)
{
    _strength = strength;
}

void SwirlField::SetStrengthFallOff(double strengthFallOff)
{
    _strengthFallOff = strengthFallOff;
}

Vector3<double> SwirlField::GetOrigin() const
{
    return _origin;
}

Vector3<double> SwirlField::GetAxisDirection() const
{
    return _axisDirection;
}

double SwirlField::GetStrength() const
{
    return _strength;
}

double SwirlField::GetStrengthFallOff() const
{
    return _strengthFallOff;
}
