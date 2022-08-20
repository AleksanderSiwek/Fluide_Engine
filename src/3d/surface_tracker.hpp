#ifndef _SURFACE_TRACKER_HPP
#define _SURFACE_TRACKER_HPP

#include "triangle_mesh.hpp"
#include "scalar_grid3d.hpp"


class SurfaceTracker
{
    public:
        SurfaceTracker();

        virtual ~SurfaceTracker();

        virtual void BuildSurface(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, TriangleMesh& mesh) = 0;
};


#endif // _SURFACE_TRACKER_HPP