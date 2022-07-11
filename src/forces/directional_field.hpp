#ifndef DIRECTIONAL_FIELD_HPP
#define DIRECTIONAL_FIELD_HPP

#include "external_force.hpp"


class DirectionalField : public ExternalForce
{
    public:
        DirectionalField(Vector3<double> strngth);
        DirectionalField(const DirectionalField& field);

        Vector3<double> Sample(const Vector3<double>& position) const override;

        void SetStrength(Vector3<double> strength);

    private:
        Vector3<double> _strength;
};

#endif //DIRECTIONAL_FIELD_HPP