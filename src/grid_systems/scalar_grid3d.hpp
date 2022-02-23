#ifndef _SCALAR_GRID3D_HPP
#define _SCALAR_GRID3D_HPP

#include "grid3d.hpp"

class ScalarGrid3D : public Grid3D, public Array3<double>
{
    public:
        ScalarGrid3D(Vector3<size_t> size=1, Vector3<double> origin=0, Vector3<double> spacing=1, double initialValue=0);
        ScalarGrid3D(const ScalarGrid3D& grid);

        ~ScalarGrid3D();

        Vector3<double> Gradient(const Vector3<double>& pos);
        double Laplacian(const Vector3<double>& pos);
};


#endif // _SCALAR_GRID3D_HPP