#ifndef _MARCHING_CUBES_SOLVER_HPP
#define _MARCHING_CUBES_SOLVER_HPP

#include "surface_tracker.hpp"


class MarchingCubesSolver : public SurfaceTracker
{
    public:
        MarchingCubesSolver();

        ~MarchingCubesSolver();

        void BuildSurface(const PointCloud& pointCloud, TriangleMesh* mesh) override;
};

#endif // _MARCHING_CUBES_SOLVER_HPP