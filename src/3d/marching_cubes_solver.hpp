#ifndef _MARCHING_CUBES_SOLVER_HPP
#define _MARCHING_CUBES_SOLVER_HPP

#include <array>

#include "surface_tracker.hpp"
#include "bounding_box_3d.hpp"

// Code adapted from: https://github.com/Magnus2/MeshReconstruction

class MarchingCubesSolver : public SurfaceTracker
{
    public:
        MarchingCubesSolver();

        ~MarchingCubesSolver();

        void BuildSurface(const ScalarGrid3D& sdf, TriangleMesh* mesh) override;
    
    private:
        double _isoValue;

        Vector3<double> Gradient(const ScalarGrid3D& sdf, int i, int j, int k);
        Vector3<double> NumGrad(const ScalarGrid3D& sdf, const Vector3<double>& pos);
        void SolveSingleCube(const ScalarGrid3D& sdf, Vector3<double> origin, Vector3<double> size, TriangleMesh* mesh);
        int CalcualteSignConfig(double* distances);
        Vector3<double> LerpVertex(int i1, int i2, Vector3<double>* positions, double* distances) const;

        static const int _signConfigToTriangles[256][16];
        static const int _edges[12][3];
        static const int _signConfigToIntersectedEdges[256];
};

#endif // _MARCHING_CUBES_SOLVER_HPP