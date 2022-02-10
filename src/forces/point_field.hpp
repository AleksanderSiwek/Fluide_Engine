#ifndef POINT_FIELD_HPP
#define POINT_FIELD_HPP

#include "../common/vector3.hpp"

class PointField
{
    public:
        PointField(Vector3<double> position, double strength, double strengthFallOff = 1);
        PointField(const PointField& field);

        ~PointField();

        void SetPosition(Vector3<double> position);
        void SetStrength(double strength);
        void SetStrengthFallOff(double strengthFallOff);

        Vector3<double> GetPosition() const;
        double GetStrength() const;
        double GetStrengthFallOff() const;

        Vector3<double> RescaleVector(Vector3<double> point, Vector3<double> value) const;

    private:
        Vector3<double> _position;
        double _strength; 
        double _strengthFallOff;

        Vector3<double> GetDistance(Vector3<double> point) const;
        Vector3<double> GetDirection(Vector3<double> point) const;
};

#endif POINT_FIELD_HPP