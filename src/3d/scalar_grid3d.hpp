#ifndef _SCALAR_GRID_3D_HPP
#define _SCALAR_GRID_3D_HPP

#include "../common/array3.hpp"
#include "../common/grid3d.hpp"
#include "../common/scalar_field3.hpp"
#include "../common/math_utils.hpp"


class ScalarGrid3D : public Array3<double>, public Grid3D, public ScalarField3
{
    public:
        ScalarGrid3D(size_t width, size_t height, size_t depth, const double& initailValue = 0, Vector3<double> origin = 0, Vector3<double> gridSpacing = 1);
        ScalarGrid3D(const Vector3<size_t>& size=1, const double& initailValue = 0, Vector3<double> origin = 0, Vector3<double> gridSpacing = 1);
        ScalarGrid3D(const ScalarGrid3D& grid);

        ~ScalarGrid3D();

        virtual double Sample(const Vector3<double>& position) const override;
        virtual Vector3<double> Gradient(const Vector3<double>& position) const override;
        virtual double Laplacian(const Vector3<double>& position) const override;

    private:
};

#endif // _SCALAR_GRID_3D_HPP