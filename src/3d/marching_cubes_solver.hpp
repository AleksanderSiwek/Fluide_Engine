#ifndef _MARCHING_CUBES_SOLVER_HPP
#define _MARCHING_CUBES_SOLVER_HPP

#include <array>
#include <unordered_map>

#include "surface_tracker.hpp"
#include "bounding_box_3d.hpp"

// Code adapted from: https://github.com/Magnus2/MeshReconstruction

class MarchingCubesSolver : public SurfaceTracker
{
    public:
        MarchingCubesSolver();

        ~MarchingCubesSolver();

        void BuildSurface(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, TriangleMesh& mesh) override;
    
        typedef size_t MarchingCubeVertexHashKey;
        typedef size_t MarchingCubeVertexId;
        typedef std::unordered_map<MarchingCubeVertexHashKey, MarchingCubeVertexId> MarchingCubeVertexMap;

    private:
        double _isoValue;

        void CalculateXYZMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, MarchingCubeVertexMap& vertexMap, TriangleMesh* mesh);
        void CalculateXYMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, TriangleMesh* mesh);
        void CalculateYZMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, TriangleMesh* mesh);
        void CalculateXZMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, TriangleMesh* mesh);
        void SolveXY(const ScalarGrid3D& sdf, Vector3<int> iter, MarchingCubeVertexMap& vertexMapBack, MarchingCubeVertexMap& vertexMapFront, TriangleMesh* mesh);
        void SolveYZ(const ScalarGrid3D& sdf, Vector3<int> iter, MarchingCubeVertexMap& vertexMapLeft, MarchingCubeVertexMap& vertexMapRight, TriangleMesh* mesh);
        void SolveXZ(const ScalarGrid3D& sdf, Vector3<int> iter, MarchingCubeVertexMap& vertexMapDown, MarchingCubeVertexMap& vertexMapUp, TriangleMesh* mesh);   

        Vector3<double> Gradient(const ScalarGrid3D& sdf, int i, int j, int k);
        size_t GlobalVertexId(size_t i, size_t j, size_t k, const Vector3<size_t>& dim, size_t localVertexId);
        size_t GlobalEdgeId(size_t i, size_t j, size_t k, const Vector3<size_t>& dim, size_t localEdgeId);
        void SolveSingleCube(const ScalarGrid3D& sdf, Vector3<int> iter, MarchingCubeVertexMap& vertexMap, TriangleMesh* mesh);
        void SolveSingleSquare(const std::array<double, 4>& data, const std::array<size_t, 8>& vertexAndEdgeIdx, 
                               const Vector3<double>& normal, const std::array<Vector3<double>, 4>& corners, 
                               MarchingCubeVertexMap& vertexMap, TriangleMesh* mesh);
        double DistanceToZeroLevelSet(double phi0, double phi1);
        bool QueryVertexId(const MarchingCubeVertexMap& vertexMap, MarchingCubeVertexHashKey vKey, MarchingCubeVertexId* vId);

        static const int _edgeConnection[12][2];
        static const int _cubeEdgeFlags[256];
        static const int _triangleConnectionTable3D[256][16];
        static const int _squareEdgeFlags[16];
        static const int _edgeConnection2D[4][2];
        static const int _triangleConnectionTable2D[16][13];
};

#endif // _MARCHING_CUBES_SOLVER_HPP