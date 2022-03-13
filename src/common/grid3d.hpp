#ifndef GRID_3D_HPP
#define GRID_3D_HPP

#include "../common/array3.hpp"


class Grid3D
{
    public:
        Grid3D(Vector3<double> origin=0, Vector3<double> gridSpacing=1) 
            :  _origin(origin), _gridSpacing(gridSpacing) 
        { }
        Grid3D(const Grid3D& grid) 
            :  _origin(grid.GetOrigin()), _gridSpacing(grid.GetGridSpacing()) 
        { }

        virtual ~Grid3D() { }

        Vector3<double> GridIndexToPosition(size_t i, size_t j, size_t k) 
        { 
            return Vector3<double>(i * _gridSpacing.x, j * _gridSpacing.y, k * _gridSpacing.z); 
        }
        Vector3<double> GridIndexToPosition(Vector3<double> position) 
        { 
            return Vector3<double>( position.x * _gridSpacing.x, 
                                    position.y * _gridSpacing.y, 
                                    position.z * _gridSpacing.z); 
        }

        void SetOrigin(Vector3<double> origin) { _origin = origin; }
        virtual void SetGridSpacing(Vector3<double> gridSpacing) { _gridSpacing = gridSpacing; }

        Vector3<double> GetOrigin() const { return _origin; }
        Vector3<double> GetGridSpacing() const { return _gridSpacing; }

    protected:
        Vector3<double> _origin;
        Vector3<double> _gridSpacing;
};

#endif // GRID_3D_HPP