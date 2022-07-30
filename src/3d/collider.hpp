#ifndef _COLLIDER_HPP
#define _COLLIDER_HPP

#include "../common/vector3.hpp"


class Collider
{
    public:
        Collider() {}
        virtual ~Collider() {}

        virtual bool IsInside(const Vector3<double>& position) = 0;
        virtual double GetClosestDistanceAt(const Vector3<double>& position) = 0;
        virtual Vector3<double> GetVelocityAt(const Vector3<double>& position) = 0;
        virtual void ResolveCollision(double radius, double restitutionCoefficient, Vector3<double>* position, Vector3<double>* velocity) = 0;
        
    private:
};

#endif // _COLLIDER_HPP