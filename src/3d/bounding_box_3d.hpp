#ifndef _BOUNDING_BOX_3D_HPP
#define _BOUNDING_BOX_3D_HPP

#include "../common/vector3.hpp"

class BoundingBox3D
{
    public:
        BoundingBox3D(Vector3<double> origin=0, Vector3<double> size=1);

        ~BoundingBox3D();

        bool IsInside(Vector3<double> point) const;

        Vector3<double> GetOrigin() const;
        Vector3<double> GetSize() const;

        void SetOrigin(Vector3<double> origin);
        void SetSize(Vector3<double> size);

    private:
        Vector3<double> _origin;
        Vector3<double> _size;
};

#endif // _BOUNDING_BOX_3D_HPP