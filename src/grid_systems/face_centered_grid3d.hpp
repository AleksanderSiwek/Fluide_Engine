#ifndef _FACE_CENTERED_GRID3D_HPP
#define _FACE_CENTERED_GRID3D_HPP

#include "../common/grid3d.hpp"


class FaceCenteredGrid3D : public Grid3D
{
    public:
        FaceCenteredGrid3D(Vector3<size_t> size=1, Vector3<double> origin=0, Vector3<double> spacing=1, Vector3<double> initalValue=0);
        FaceCenteredGrid3D(const FaceCenteredGrid3D& grid);

        ~FaceCenteredGrid3D();

        void Resize(Vector3<size_t> size);
        void Fill(double xVal, double yVal, double zVal);
        void Fill(const Array3<double>& dataX, const Array3<double>& dataY, const Array3<double>& dataZ);
        bool IsEqual(const FaceCenteredGrid3D& grid);
        double& x(size_t i, size_t j, size_t k);
        const double& x(size_t i, size_t j, size_t k) const; 
        double& y(size_t i, size_t j, size_t k);
        const double& y(size_t i, size_t j, size_t k) const; 
        double& z(size_t i, size_t j, size_t k);
        const double& z(size_t i, size_t j, size_t k) const; 
        Vector3<double> ValueAtCellCenter(size_t i, size_t j, size_t k) const;
        double DivergenceAtCallCenter(size_t i, size_t j, size_t k) const;
        Vector3<double> CurlAtCellCentre(size_t i, size_t j, size_t k) const;

        void SetGridSpacing(Vector3<double> gridSpacing) override;

        Vector3<size_t> GetSize() const;
        Vector3<size_t> GetActualSize() const;
        Vector3<double> GetDataXOrigin() const;
        Vector3<double> GetDataYOrigin() const;
        Vector3<double> GetDataZOrigin() const;
        Array3<double> GetDataX() const;
        Array3<double> GetDataY() const;
        Array3<double> GetDataZ() const;
        Array3<double>* GetDataXPtr();
        Array3<double>* GetDataYPtr();
        Array3<double>* GetDataZPtr();
        const Array3<double>& GetDataXRef() const;
        const Array3<double>& GetDataYRef() const;
        const Array3<double>& GetDataZRef() const;

    private:
        Vector3<double> _dataXOrigin;
        Vector3<double> _dataYOrigin;
        Vector3<double> _dataZOrigin;

        Array3<double> _dataX;
        Array3<double> _dataY;
        Array3<double> _dataZ;

        Vector3<size_t> _size;

        void CalculateDataOrigins();
        void SetSize(Vector3<size_t> size);
};

#endif // _FACE_CENTERED_GRID3D_HPP