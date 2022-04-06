#ifndef DIRECTIONAL_FIELD_HPP
#define DIRECTIONAL_FIELD_HPP

#include "external_force.hpp"
#include "../common/vector_field3.hpp"


class DirectionalField : public VectorField3
{
    public:
        DirectionalField(Vector3<double> strngth);
        DirectionalField(const DirectionalField& field);

        Vector3<double> Sample(const Vector3<double>& position) const override;
        Vector3<double> Divergence(const Vector3<double>& position) const override;
        Vector3<double> Curl(const Vector3<double>& position) const override;

        void SetStrength(Vector3<double> strength);

    private:
        Vector3<double> _strength;
};

#endif //DIRECTIONAL_FIELD_HPP