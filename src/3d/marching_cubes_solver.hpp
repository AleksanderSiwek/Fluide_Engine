#ifndef _MARCHING_CUBES_SOLVER_HPP
#deifne _MARCHING_CUBES_SOLVER_HPP

#include "surface_tracker.hpp"

class MarchingCubesSolver
{
    public:
        MarchingCubesSolver();

        ~MarchingCubesSolver();

        void BuildSurface(const PointCloud& pointCloud, TriangleMesh* mesh) override;
};

#endif // _MARCHING_CUBES_SOLVER_HPP