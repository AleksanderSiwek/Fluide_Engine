#ifndef _FACE_CENTERED_GRID3D_HPP
#define _FACE_CENTERED_GRID3D_HPP

#include "grid3d.hpp"

class FaceCenteredGrid3D : public Grid3D<Vector3<double>>
{
    public:
        FaceCenteredGrid3D(Vector3<size_t> size=1, Vector3<double> origin=0, Vector3<double> spacing=1, Vector3<double> initialValue=0);
        FaceCenteredGrid3D(const FaceCenteredGrid3D& grid);

        ~FaceCenteredGrid3D();

        double& x(size_t i, size_t j, size_t k);
        const double& x(size_t i, size_t j, size_t k) const; 
        double& y(size_t i, size_t j, size_t k);
        const double& y(size_t i, size_t j, size_t k) const; 
        double& z(size_t i, size_t j, size_t k);
        const double& z(size_t i, size_t j, size_t k) const; 

        void SetGridSpacing(Vector3<double> gridSpacing) override;

        Vector3<size_t> GetSize() const override;
        Vector3<size_t> GetActualSize() const;
        Vector3<double> GetDataXOrigin() const;
        Vector3<double> GetDataYOrigin() const;
        Vector3<double> GetDataZOrigin() const;

    private:
        Vector3<double> _dataXOrigin;
        Vector3<double> _dataYOrigin;
        Vector3<double> _dataZOrigin;

        void CalculateDataOrigins();

        void SetSize(Vector3<size_t> size) override;
};

#endif // _FACE_CENTERED_GRID3D_HPP