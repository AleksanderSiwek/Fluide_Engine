#ifndef _SURFACE_TRACKER_HPP
#define _SURFACE_TRACKER_HPP

#include "triangle_mesh.hpp"

typedef std::vector<Vector3<double>> PointCloud;


class SurfaceTracker
{
    public:
        SurfaceTracker();

        virtual ~SurfaceTracker();

        virtual void BuildSurface(const PointCloud& pointCloud, TriangleMesh* mesh) = 0;
};


#endif // _SURFACE_TRACKER_HPP