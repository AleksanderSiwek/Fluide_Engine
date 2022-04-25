#ifndef GRID_3D_HPP
#define GRID_3D_HPP

#include "../common/vector3.hpp"
#include "../common/math_utils.hpp"


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

        Vector3<double> GridIndexToPosition(size_t i, size_t j, size_t k) const
        { 
            return Vector3<double>(_gridSpacing.x + i * _gridSpacing.x, _gridSpacing.y + j * _gridSpacing.y, _gridSpacing.z + k * _gridSpacing.z); 
        }
        Vector3<double> GridIndexToPosition(Vector3<size_t> position) const
        { 
            return GridIndexToPosition(position.x, position.y, position.z);
        }

        Vector3<size_t> PositionToGridIndex(Vector3<double> position) const
        {
            Vector3<double> inverserSpacing(1 / _gridSpacing.x, 1 / _gridSpacing.y, 1 / _gridSpacing.z);
            return Vector3<size_t>( (size_t)floor(position.x * inverserSpacing.x),
                                    (size_t)floor(position.y * inverserSpacing.y),
                                    (size_t)floor(position.z * inverserSpacing.z));
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