#ifndef DIRECTIONAL_FIELD_HPP
#define DIRECTIONAL_FIELD_HPP

#include "../common/vector3.hpp"

class DirectionalField
{
    public:
        DirectionalField(Vector3<double> direction, double strength);
        DirectionalField(const DirectionalField& field);

        void SetDirection(Vector3<double> direction);
        void SetStrength(double strength);

        Vector3<double> GetDirection() const;
        double GetStrength() const;

        Vector3<double> RescaleVector(Vector3<double> value);

    private:
        Vector3<double> _direction;
        double _strength;
};

#endif //DIRECTIONAL_FIELD_HPP