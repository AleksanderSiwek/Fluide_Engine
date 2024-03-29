#ifndef POINT_FIELD_HPP
#define POINT_FIELD_HPP

#include "external_force.hpp"


class PointField : public ExternalForce
{
    public:
        PointField(Vector3<double> origin = 0, double strength = 0, double strengthFallOff = 1);
        PointField(const PointField& field);

        ~PointField();

        Vector3<double> Sample(const Vector3<double>& position) const override;

        void SetPosition(Vector3<double> origin);
        void SetStrength(double strength);
        void SetStrengthFallOff(double strengthFallOff);

        Vector3<double> GetOrigin() const;
        double GetStrength() const;
        double GetStrengthFallOff() const;

    private:
        Vector3<double> _origin;
        double _strength; 
        double _strengthFallOff;

        double GetDistance(Vector3<double> point) const;
        Vector3<double> GetDirection(Vector3<double> point) const;
};

#endif POINT_FIELD_HPP