#ifndef GRID_3D_HPP
#define GRID_3D_HPP

#include "../common/array3.hpp"

template <typename T>
class Grid3D : public Array3<T>
{
    public:
        Grid3D(Vector3<size_t> size=1, Vector3<double> origin=0, Vector3<double> gridSpacing=1, T initialValue=0) 
            : Array3<T>(size, initialValue), _origin(origin), _gridSpacing(gridSpacing) { }
        Grid3D(const Grid3D& grid) 
            : Array3(grid.GetSize()), _origin(grid.GetOrigin()), _spacing(grid.GetSpacing()) 
        { 
            Fill(grid);
        }

        virtual ~Grid3D() { }

        void SetOrigin(Vector3<double> origin) { _origin = origin; }
        virtual void SetGridSpacing(Vector3<double> gridSpacing) { _gridSpacing = gridSpacing; }

        Vector3<double> GetOrigin() const { return _origin; }
        Vector3<double> GetGridSpacing() const { return _gridSpacing; }
        Vector3<double> GetDiemensions() const { return GetSize() * _gridSpacing; }

    protected:
        Vector3<double> _origin;
        Vector3<double> _gridSpacing;
};

#endif // GRID_3D_HPP