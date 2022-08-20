#ifndef _VOLUME_FORCE_HPP
#define _VOLUME_FORCE_HPP

#include "external_force.hpp"
#include "../common/array3.hpp"
#include "../3d/triangle_mesh.hpp"
#include "../3d/scalar_grid3d.hpp"
#include "../3d/bounding_box_3d.hpp"


class VolumeField : public ExternalForce
{
    public:
        VolumeField(const TriangleMesh& mesh, Vector3<size_t> resolution, BoundingBox3D domain, double strength);

        ~VolumeField();

        Vector3<double> Sample(const Vector3<double>& position) const override;

        void SetStrength(double strength);

        double GetStrength() const;

    private:
        Vector3<double> _origin;
        Vector3<double> _gridSpacing;
        Vector3<size_t> _resolution;
        ScalarGrid3D _sdf;
        double _strength; 
        TriangleMesh _mesh;
        Array3<Vector3<double>> _vectorField;

        void Initialize();
};

#endif // _VOLUME_FORCE_HPP