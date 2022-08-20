#ifndef _SWIRL_FIELD_HPP
#define _SWIRL_FIELD_HPP


#include "external_force.hpp"


class SwirlField : public ExternalForce
{
    public:
        SwirlField(Vector3<double> origin = 0, Vector3<double> axisDirection = Vector3<double>(0, 1, 0), double strength = 0, double strengthFallOff = 1);
        SwirlField(const SwirlField& field);

        ~SwirlField();

        Vector3<double> Sample(const Vector3<double>& position) const override;

        void SetOrigin(Vector3<double> origin);
        void SetAxisDirection(Vector3<double> axisDirection);
        void SetStrength(double strength);
        void SetStrengthFallOff(double strengthFallOff);

        Vector3<double> GetOrigin() const;
        Vector3<double> GetAxisDirection() const;
        double GetStrength() const;
        double GetStrengthFallOff() const;

    private:
        Vector3<double> _origin;
        Vector3<double> _axisDirection;
        double _strength; 
        double _strengthFallOff;
};

#endif // _SWIRL_FIELD_HPP