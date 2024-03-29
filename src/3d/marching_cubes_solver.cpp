#include "marching_cubes_solver.hpp"

inline Vector3<double> SafeNormalize(Vector3<double>& n) 
{
    if (n.GetLength() * n.GetLength() > 0.0)
     {
        n.Normalize();
        return n;
    } else {
        return n;
    }
}

MarchingCubesSolver::MarchingCubesSolver() 
    : _isoValue(0.0)
{

}

MarchingCubesSolver::~MarchingCubesSolver()
{

}

void MarchingCubesSolver::BuildSurface(const ScalarGrid3D& fluidSdf, const ScalarGrid3D& colliderSdf, TriangleMesh& mesh)
{
    MarchingCubeVertexMap vertexMap;
    const auto& size = fluidSdf.GetSize();
    ScalarGrid3D sdf = fluidSdf;

    sdf.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if(fluidSdf(i, j, k) < 0 && colliderSdf(i, j, k) < 0)
        {
            sdf(i, j, k) = colliderSdf(i, j, k) * (-1);
        }
    });
    // Vector3<double> scaleFactor(2, 2, 2);
    // const auto& newSize = Vector3<size_t>((size_t)(size.x * scaleFactor.x), (size_t)(size.y * scaleFactor.y), (size_t)(size.z * scaleFactor.z));
    // ScalarGrid3D rescaledSdf(newSize, 0, origin, gridSpacing / scaleFactor);
    // rescaledSdf.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    // {
    //     rescaledSdf(i, j, k) = sdf.Sample(rescaledSdf.GridIndexToPosition(i, j, k));
    // });


    CalculateXYZMeshPart(sdf, size, vertexMap, &mesh);
    CalculateXYMeshPart(sdf, size, &mesh);
    CalculateYZMeshPart(sdf, size, &mesh);
    CalculateXZMeshPart(sdf, size, &mesh);
}

void MarchingCubesSolver::CalculateXYZMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, MarchingCubeVertexMap& vertexMap, TriangleMesh* mesh)
{
    int numX = static_cast<int>(size.x);
    int numY = static_cast<int>(size.y);
    int numZ = static_cast<int>(size.z); 

    for(int i = 0; i < numX - 1; i++)
    {
        for(int j = 0; j < numY - 1; j++)
        {
            for(int k = 0; k < numZ - 1; k++)
            {
                Vector3<int> iter(i, j, k);
                SolveSingleCube(sdf, iter, vertexMap, mesh);
            }
        }
    }
}

void MarchingCubesSolver::CalculateXYMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, TriangleMesh* mesh)
{
    MarchingCubeVertexMap vertexMapBack;
    MarchingCubeVertexMap vertexMapFront;

    for(int j = 0; j < size.y - 1; j++)
    {
        for(int i = 0; i < size.x - 1; i++)
        {
            Vector3<int> iter(i, j, 0);
            SolveXY(sdf, iter, vertexMapBack, vertexMapFront, mesh);
        }
    }
}

void MarchingCubesSolver::CalculateYZMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, TriangleMesh* mesh)
{
    MarchingCubeVertexMap vertexMapLeft;
    MarchingCubeVertexMap vertexMapRight;

    for(int k = 0; k < size.z - 1; k++)
    {
        for(int j = 0; j < size.y - 1; j++)
        {
            Vector3<int> iter(0, j, k);
            SolveYZ(sdf, iter, vertexMapLeft, vertexMapRight, mesh);
        }
    }
}

void MarchingCubesSolver::CalculateXZMeshPart(const ScalarGrid3D& sdf, const Vector3<size_t>& size, TriangleMesh* mesh)
{
    MarchingCubeVertexMap vertexMapDown;
    MarchingCubeVertexMap vertexMapUp;

    for(int k = 0; k < size.z - 1; k++)
    {
        for(int i = 0; i < size.x - 1; i++)
        {
            Vector3<int> iter(i, 0, k);
            SolveXZ(sdf, iter, vertexMapDown, vertexMapUp, mesh);
        }
    }
}

Vector3<double> MarchingCubesSolver::Gradient(const ScalarGrid3D& sdf, int i, int j, int k)
{
    Vector3<double> ret;
    const auto& dim = sdf.GetSize();
    const auto& gridSpacing = sdf.GetGridSpacing();
    const Vector3<double> invGridSpacing = 1.0 / gridSpacing;

    int ip = i + 1;
    int im = i - 1;
    int jp = j + 1;
    int jm = j - 1;
    int kp = k + 1;
    int km = k - 1;
    int dimx = static_cast<int>(dim.x);
    int dimy = static_cast<int>(dim.y);
    int dimz = static_cast<int>(dim.z);
    if (i > dimx - 2) 
    {
        ip = i;
    } 
    else if (i == 0) 
    {
        im = 0;
    }
    if (j > dimy - 2) 
    {
        jp = j;
    } 
    else if (j == 0) 
    {
        jm = 0;
    }
    if (k > dimz - 2) 
    {
        kp = k;
    } 
    else if (k == 0) 
    {
        km = 0;
    }
    ret.x = 0.5f * invGridSpacing.x * (sdf(ip, j, k) - sdf(im, j, k));
    ret.y = 0.5f * invGridSpacing.y * (sdf(i, jp, k) - sdf(i, jm, k));
    ret.z = 0.5f * invGridSpacing.z * (sdf(i, j, kp) - sdf(i, j, km));
    return ret;
}

size_t MarchingCubesSolver::GlobalVertexId(size_t i, size_t j, size_t k, const Vector3<size_t>& dim, size_t localVertexId)
{
    static const int vertexOffset3D[8][3] = {{0, 0, 0}, {2, 0, 0}, {2, 0, 2},
                                            {0, 0, 2}, {0, 2, 0}, {2, 2, 0},
                                            {2, 2, 2}, {0, 2, 2}};

    return ((2 * k + vertexOffset3D[localVertexId][2]) * 2 * dim.y +
        (2 * j + vertexOffset3D[localVertexId][1])) *
            2 * dim.x +
        (2 * i + vertexOffset3D[localVertexId][0]);
}

size_t MarchingCubesSolver::GlobalEdgeId(size_t i, size_t j, size_t k, const Vector3<size_t>& dim, size_t localEdgeId) 
{
    // See edgeConnection in marching_cubes_table.h for the edge ordering.
    static const int edgeOffset3D[12][3] = {
        {1, 0, 0}, {2, 0, 1}, {1, 0, 2}, {0, 0, 1}, {1, 2, 0}, {2, 2, 1},
        {1, 2, 2}, {0, 2, 1}, {0, 1, 0}, {2, 1, 0}, {2, 1, 2}, {0, 1, 2}};

    return ((2 * k + edgeOffset3D[localEdgeId][2]) * 2 * dim.y +
            (2 * j + edgeOffset3D[localEdgeId][1])) *
               2 * dim.x +
           (2 * i + edgeOffset3D[localEdgeId][0]);
}

void MarchingCubesSolver::SolveSingleCube(const ScalarGrid3D& sdf, Vector3<int> iter, MarchingCubeVertexMap& vertexMap, TriangleMesh* mesh)
{
    std::array<double, 8> data;
    std::array<size_t, 12> edgeIds;
    std::array<Vector3<double>, 8> normals;
    BoundingBox3D bbox;

    const int& i = iter.x;
    const int& j = iter.y;
    const int& k = iter.z;

    data[0] = sdf(i, j, k);
    data[1] = sdf(i + 1, j, k);
    data[4] = sdf(i, j + 1, k);
    data[5] = sdf(i + 1, j + 1, k);
    data[3] = sdf(i, j, k + 1);
    data[2] = sdf(i + 1, j, k + 1);
    data[7] = sdf(i, j + 1, k + 1);
    data[6] = sdf(i + 1, j + 1, k + 1);

    normals[0] = Gradient(sdf, i, j, k);
    normals[1] = Gradient(sdf, i + 1, j, k);
    normals[4] = Gradient(sdf, i, j + 1, k);
    normals[5] = Gradient(sdf, i + 1, j + 1, k);
    normals[3] = Gradient(sdf, i, j, k + 1);
    normals[2] = Gradient(sdf, i + 1, j, k + 1);
    normals[7] = Gradient(sdf, i, j + 1, k + 1);
    normals[6] = Gradient(sdf, i + 1, j + 1, k + 1);

    for(int e = 0; e < 12; e++)
    {
        edgeIds[e] = GlobalEdgeId(i, j, k, sdf.GetSize(), e);
    }

    bbox.SetOrigin(sdf.GridIndexToPosition(i, j, k));
    bbox.SetSize(sdf.GetGridSpacing());

    int itrVertex, itrEdge, itrTri;
    int idxFlagSize = 0, idxEdgeFlags = 0;
    int idxVertexOfTheEdge[2];

    Vector3<double> pos, pos0, pos1, normal, normal0, normal1;
    double phi0, phi1;
    double alpha;
    Vector3<double> e[12], n[12];

    for(itrVertex = 0; itrVertex < 8; itrVertex++)
    {
        if(data[itrVertex] <= _isoValue)
        {
            idxFlagSize |= 1 << itrVertex;
        }
    }

    if(idxFlagSize == 0 || idxFlagSize == 255)
    {
        return;
    }

    idxEdgeFlags = _cubeEdgeFlags[idxFlagSize];

    for(itrEdge = 0; itrEdge < 12; itrEdge++)
    {
        if(idxEdgeFlags & (1 << itrEdge))
        {
            idxVertexOfTheEdge[0] = _edgeConnection[itrEdge][0];
            idxVertexOfTheEdge[1] = _edgeConnection[itrEdge][1];

            // cube vertex ordering to x-major ordering
            static int indexMap[8] = {0, 1, 5, 4, 2, 3, 7, 6};

            // Find the phi = 0 position
            pos0 = bbox.Corner(indexMap[idxVertexOfTheEdge[0]]);
            pos1 = bbox.Corner(indexMap[idxVertexOfTheEdge[1]]);

            normal0 = normals[idxVertexOfTheEdge[0]];
            normal1 = normals[idxVertexOfTheEdge[1]];

            phi0 = data[idxVertexOfTheEdge[0]] - _isoValue;
            phi1 = data[idxVertexOfTheEdge[1]] - _isoValue;

            alpha = DistanceToZeroLevelSet(phi0, phi1);

            if (alpha < 0.000001) {
                alpha = 0.000001;
            }
            if (alpha > 0.999999) {
                alpha = 0.999999;
            }

            pos = (1.0 - alpha) * pos0 + alpha * pos1;
            normal = (1.0 - alpha) * normal0 + alpha * normal1;

            e[itrEdge] = pos;
            n[itrEdge] = normal;
        }
    }

    // Build triangles
    for(itrTri = 0; itrTri < 5; itrTri++)
    {
        if(_triangleConnectionTable3D[idxFlagSize][3 * itrTri] < 0)
        {
            break;
        }

        size_t face[3];
        for(int vert = 0; vert < 3; vert++)
        {
            int triKey = 3 * itrTri + vert;
            MarchingCubeVertexHashKey vKey = edgeIds[_triangleConnectionTable3D[idxFlagSize][triKey]];
            MarchingCubeVertexId vId;
            if(QueryVertexId(vertexMap, vKey, &vId))
            {
                face[vert] = vId;
            }
            else
            {
                face[vert] = mesh->GetVerticies().size();
                mesh->AddNormal(SafeNormalize(n[_triangleConnectionTable3D[idxFlagSize][triKey]]));
                mesh->AddVertex(e[_triangleConnectionTable3D[idxFlagSize][triKey]]);
                vertexMap.insert(std::make_pair(vKey, face[vert]));
            }
        }
        mesh->AddTriangle(Triangle3D_t(face[0], face[1], face[2], mesh->GetNormals().size() - 1));
    }
}

void MarchingCubesSolver::SolveXY(const ScalarGrid3D& sdf, Vector3<int> iter, MarchingCubeVertexMap& vertexMapBack, MarchingCubeVertexMap& vertexMapFront, TriangleMesh* mesh)
{ 
    const auto& size = sdf.GetSize();
    std::array<double, 4> data;
    std::array<size_t, 8> vertexAndEdgeIdx;
    std::array<Vector3<double>, 4> corners;
    Vector3<double> normal;
    BoundingBox3D bbox;

    int i = iter.x;
    int j = iter.y;
    
    // Back
    int k = 0;
    normal = Vector3<double>(0, 0, -1);

    data[0] = sdf(i + 1, j, k);
    data[1] = sdf(i, j, k);
    data[2] = sdf(i, j + 1, k);
    data[3] = sdf(i + 1, j + 1, k);
    vertexAndEdgeIdx[0] = GlobalVertexId(i, j, k, size, 1);
    vertexAndEdgeIdx[1] = GlobalVertexId(i, j, k, size, 0);
    vertexAndEdgeIdx[2] = GlobalVertexId(i, j, k, size, 4);
    vertexAndEdgeIdx[3] = GlobalVertexId(i, j, k, size, 5);
    vertexAndEdgeIdx[4] = GlobalEdgeId(i, j, k, size, 0);
    vertexAndEdgeIdx[5] = GlobalEdgeId(i, j, k, size, 8);
    vertexAndEdgeIdx[6] = GlobalEdgeId(i, j, k, size, 4);
    vertexAndEdgeIdx[7] = GlobalEdgeId(i, j, k, size, 9);
    corners[0] = sdf.GridIndexToPosition(i + 1, j, k);
    corners[1] = sdf.GridIndexToPosition(i , j, k);
    corners[2] = sdf.GridIndexToPosition(i, j + 1, k);
    corners[3] = sdf.GridIndexToPosition(i + 1, j + 1, k);

    SolveSingleSquare(data, vertexAndEdgeIdx, normal, corners, vertexMapBack, mesh);

    // Front
    k = (int)size.z - 2;
    normal = Vector3<double>(0, 0, 1);

    data[0] = sdf(i, j, k + 1);
    data[1] = sdf(i + 1, j, k + 1);
    data[2] = sdf(i + 1, j + 1, k + 1);
    data[3] = sdf(i, j + 1, k + 1);
    vertexAndEdgeIdx[0] = GlobalVertexId(i, j, k, size, 3);
    vertexAndEdgeIdx[1] = GlobalVertexId(i, j, k, size, 2);
    vertexAndEdgeIdx[2] = GlobalVertexId(i, j, k, size, 6);
    vertexAndEdgeIdx[3] = GlobalVertexId(i, j, k, size, 7);
    vertexAndEdgeIdx[4] = GlobalEdgeId(i, j, k, size, 2);
    vertexAndEdgeIdx[5] = GlobalEdgeId(i, j, k, size, 10);
    vertexAndEdgeIdx[6] = GlobalEdgeId(i, j, k, size, 6);
    vertexAndEdgeIdx[7] = GlobalEdgeId(i, j, k, size, 11);
    corners[0] = sdf.GridIndexToPosition(i, j, k + 1);
    corners[1] = sdf.GridIndexToPosition(i + 1, j, k + 1);
    corners[2] = sdf.GridIndexToPosition(i + 1, j + 1, k + 1);
    corners[3] = sdf.GridIndexToPosition(i, j + 1, k + 1);
    
    SolveSingleSquare(data, vertexAndEdgeIdx, normal, corners, vertexMapFront, mesh);

}

void MarchingCubesSolver::SolveYZ(const ScalarGrid3D& sdf, Vector3<int> iter, MarchingCubeVertexMap& vertexMapLeft, MarchingCubeVertexMap& vertexMapRight, TriangleMesh* mesh)
{
    const auto& size = sdf.GetSize();
    std::array<double, 4> data;
    std::array<size_t, 8> vertexAndEdgeIdx;
    std::array<Vector3<double>, 4> corners;
    Vector3<double> normal;
    BoundingBox3D bbox;

    int j = iter.y;
    int k = iter.z;

    // Left
    int i = 0;
    normal = Vector3<double>(-1, 0, 0);

    data[0] = sdf(i, j, k);
    data[1] = sdf(i, j, k + 1);
    data[2] = sdf(i, j + 1, k + 1);
    data[3] = sdf(i, j + 1, k);
    vertexAndEdgeIdx[0] = GlobalVertexId(i, j, k, size, 0);
    vertexAndEdgeIdx[1] = GlobalVertexId(i, j, k, size, 3);
    vertexAndEdgeIdx[2] = GlobalVertexId(i, j, k, size, 7);
    vertexAndEdgeIdx[3] = GlobalVertexId(i, j, k, size, 4);
    vertexAndEdgeIdx[4] = GlobalEdgeId(i, j, k, size, 3);
    vertexAndEdgeIdx[5] = GlobalEdgeId(i, j, k, size, 11);
    vertexAndEdgeIdx[6] = GlobalEdgeId(i, j, k, size, 7);
    vertexAndEdgeIdx[7] = GlobalEdgeId(i, j, k, size, 8);
    corners[0] = sdf.GridIndexToPosition(i, j, k);
    corners[1] = sdf.GridIndexToPosition(i, j, k + 1);
    corners[2] = sdf.GridIndexToPosition(i, j + 1, k + 1);
    corners[3] = sdf.GridIndexToPosition(i, j + 1, k);

    SolveSingleSquare(data, vertexAndEdgeIdx, normal, corners, vertexMapLeft, mesh);

    // Right
    i = (int)size.x - 2;
    normal = Vector3<double>(1, 0, 0);

    data[0] = sdf(i + 1, j, k + 1);
    data[1] = sdf(i + 1, j, k);
    data[2] = sdf(i + 1, j + 1, k);
    data[3] = sdf(i + 1, j + 1, k + 1);
    vertexAndEdgeIdx[0] = GlobalVertexId(i, j, k, size, 2);
    vertexAndEdgeIdx[1] = GlobalVertexId(i, j, k, size, 1);
    vertexAndEdgeIdx[2] = GlobalVertexId(i, j, k, size, 5);
    vertexAndEdgeIdx[3] = GlobalVertexId(i, j, k, size, 6);
    vertexAndEdgeIdx[4] = GlobalEdgeId(i, j, k, size, 1);
    vertexAndEdgeIdx[5] = GlobalEdgeId(i, j, k, size, 9);
    vertexAndEdgeIdx[6] = GlobalEdgeId(i, j, k, size, 5);
    vertexAndEdgeIdx[7] = GlobalEdgeId(i, j, k, size, 10);
    corners[0] = sdf.GridIndexToPosition(i + 1, j, k + 1);
    corners[1] = sdf.GridIndexToPosition(i + 1, j, k);
    corners[2] = sdf.GridIndexToPosition(i + 1, j + 1, k);
    corners[3] = sdf.GridIndexToPosition(i + 1, j + 1, k + 1);
    
    SolveSingleSquare(data, vertexAndEdgeIdx, normal, corners, vertexMapRight, mesh);
}

void MarchingCubesSolver::SolveXZ(const ScalarGrid3D& sdf, Vector3<int> iter,MarchingCubeVertexMap& vertexMapDown, MarchingCubeVertexMap& vertexMapUp, TriangleMesh* mesh)
{
    const auto& size = sdf.GetSize();
    std::array<double, 4> data;
    std::array<size_t, 8> vertexAndEdgeIdx;
    std::array<Vector3<double>, 4> corners;
    Vector3<double> normal;
    BoundingBox3D bbox;

    int i = iter.x;
    int k = iter.z;

    // Down
    int j = 0;
    normal = Vector3<double>(0, -1, 0);

    data[0] = sdf(i, j, k);
    data[1] = sdf(i + 1, j, k);
    data[2] = sdf(i + 1, j, k + 1);
    data[3] = sdf(i, j, k + 1);
    vertexAndEdgeIdx[0] = GlobalVertexId(i, j, k, size, 0);
    vertexAndEdgeIdx[1] = GlobalVertexId(i, j, k, size, 1);
    vertexAndEdgeIdx[2] = GlobalVertexId(i, j, k, size, 2);
    vertexAndEdgeIdx[3] = GlobalVertexId(i, j, k, size, 3);
    vertexAndEdgeIdx[4] = GlobalEdgeId(i, j, k, size, 0);
    vertexAndEdgeIdx[5] = GlobalEdgeId(i, j, k, size, 1);
    vertexAndEdgeIdx[6] = GlobalEdgeId(i, j, k, size, 2);
    vertexAndEdgeIdx[7] = GlobalEdgeId(i, j, k, size, 3);
    corners[0] = sdf.GridIndexToPosition(i, j, k);
    corners[1] = sdf.GridIndexToPosition(i + 1, j, k);
    corners[2] = sdf.GridIndexToPosition(i + 1, j, k + 1);
    corners[3] = sdf.GridIndexToPosition(i, j, k + 1);

    SolveSingleSquare(data, vertexAndEdgeIdx, normal, corners, vertexMapDown, mesh);

    // Up
    j = (int)size.y - 2;
    normal = Vector3<double>(0, 1, 0);

    data[0] = sdf(i + 1, j + 1, k);
    data[1] = sdf(i, j + 1, k);
    data[2] = sdf(i, j + 1, k + 1);
    data[3] = sdf(i + 1, j + 1, k + 1);
    vertexAndEdgeIdx[0] = GlobalVertexId(i, j, k, size, 5);
    vertexAndEdgeIdx[1] = GlobalVertexId(i, j, k, size, 4);
    vertexAndEdgeIdx[2] = GlobalVertexId(i, j, k, size, 7);
    vertexAndEdgeIdx[3] = GlobalVertexId(i, j, k, size, 6);
    vertexAndEdgeIdx[4] = GlobalEdgeId(i, j, k, size, 4);
    vertexAndEdgeIdx[5] = GlobalEdgeId(i, j, k, size, 7);
    vertexAndEdgeIdx[6] = GlobalEdgeId(i, j, k, size, 6);
    vertexAndEdgeIdx[7] = GlobalEdgeId(i, j, k, size, 5);
    corners[0] = sdf.GridIndexToPosition(i + 1, j + 1, k);
    corners[1] = sdf.GridIndexToPosition(i, j + 1, k);
    corners[2] = sdf.GridIndexToPosition(i, j + 1, k + 1);
    corners[3] = sdf.GridIndexToPosition(i + 1, j + 1, k + 1);
    
    SolveSingleSquare(data, vertexAndEdgeIdx, normal, corners, vertexMapUp, mesh);
}

void MarchingCubesSolver::SolveSingleSquare(const std::array<double, 4>& data, const std::array<size_t, 8>& vertexAndEdgeIdx, 
                                            const Vector3<double>& normal, const std::array<Vector3<double>, 4>& corners, 
                                            MarchingCubeVertexMap& vertexMap, TriangleMesh* mesh)
{
    int itrVertex, itrEdge, itrTri;
    int idxFlags = 0, idxEdgeFlags = 0;
    int idxVertexOfTheEdge[2];

    double phi0, phi1, alpha;
    Vector3<double> pos, pos0, pos1;
    Vector3<double> e[4];

    for (itrVertex = 0; itrVertex < 4; itrVertex++) {
        if (data[itrVertex] <= _isoValue) {
            idxFlags |= 1 << itrVertex;
        }
    }

    if (idxFlags == 0) {
        return;
    }

    idxEdgeFlags = _squareEdgeFlags[idxFlags];

    for (itrEdge = 0; itrEdge < 4; itrEdge++) {
        if (idxEdgeFlags & (1 << itrEdge)) {
            idxVertexOfTheEdge[0] = _edgeConnection2D[itrEdge][0];
            idxVertexOfTheEdge[1] = _edgeConnection2D[itrEdge][1];

            pos0 = corners[idxVertexOfTheEdge[0]];
            pos1 = corners[idxVertexOfTheEdge[1]];

            phi0 = data[idxVertexOfTheEdge[0]] - _isoValue;
            phi1 = data[idxVertexOfTheEdge[1]] - _isoValue;

            if (std::fabs(phi0) + std::fabs(phi1) > 1e-12) {
                alpha = std::fabs(phi0) / (std::fabs(phi0) + std::fabs(phi1));
            } else {
                alpha = 0.5f;
            }

            if (alpha < 0.000001f) {
                alpha = 0.000001f;
            }
            if (alpha > 0.999999f) {
                alpha = 0.999999f;
            }

            pos = ((1.f - alpha) * pos0 + alpha * pos1);

            e[itrEdge] = pos;
        }
    }

    for (itrTri = 0; itrTri < 4; ++itrTri) 
    {
        if (_triangleConnectionTable2D[idxFlags][3 * itrTri] < 0) {
            break;
        }

        size_t face[3];

        for (int j = 0; j < 3; ++j) 
        {
            int idxVertex = _triangleConnectionTable2D[idxFlags][3 * itrTri + j];

            MarchingCubeVertexHashKey vKey = vertexAndEdgeIdx[idxVertex];
            MarchingCubeVertexId vId;
            if (QueryVertexId(vertexMap, vKey, &vId)) 
            {
                face[j] = vId;
            } 
            else 
            {
                face[j] = mesh->GetVerticies().size();
                mesh->AddNormal(normal);
                if (idxVertex < 4) 
                {
                    mesh->AddVertex(corners[idxVertex]);
                } 
                else 
                {
                    mesh->AddVertex(e[idxVertex - 4]);
                }
                vertexMap.insert(std::make_pair(vKey, face[j]));
            }
        }
        mesh->AddTriangle(Triangle3D_t(face[0], face[1], face[2], mesh->GetNormals().size() - 1));
    }
}

double MarchingCubesSolver::DistanceToZeroLevelSet(double phi0, double phi1) 
{
    if (std::fabs(phi0) + std::fabs(phi1) > 0.000000001) 
    {
        return std::fabs(phi0) / (std::fabs(phi0) + std::fabs(phi1));
    } 
    else 
    {
        return static_cast<double>(0.5);
    }
}

bool MarchingCubesSolver::QueryVertexId(const MarchingCubeVertexMap& vertexMap, MarchingCubeVertexHashKey vKey, MarchingCubeVertexId* vId)
{
    auto vItr = vertexMap.find(vKey);
    if (vItr != vertexMap.end()) 
    {
        *vId = vItr->second;
        return true;
    } 
    else 
    {
        return false;
    }  
}

const int MarchingCubesSolver::_edgeConnection[12][2] = {
    {0, 1}, {1, 2}, {3, 2}, {0, 3},
    {4, 5}, {5, 6}, {7, 6}, {4, 7},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

const int MarchingCubesSolver::_cubeEdgeFlags[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

const int MarchingCubesSolver::_triangleConnectionTable3D[256][16] = {                 // EdgeFlag, NodeFlag   special case for MDC
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x000,   0000 0000   X
    {  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x109,   0000 0001   O   {  0, 8,  8, 3,  3, 0
    {  0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x203,   0000 0010   O   {  0, 1,  1, 9,  9, 0
    {  1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x30a,   0000 0011   O   {  1, 9,  9, 8,  8, 3,  3, 1
    {  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x406,   0000 0100   O   {  1, 2,  2, 10, 10, 1
    {  0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x50f,   0000 0101   X
    {  9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x605,   0000 0110   O   {  0, 2,  2, 10, 10, 9,  9, 0
    {  2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1 }, // 0x70c,   0000 0111   O   {  3, 2,  2, 10, 10, 9,  9, 8,  8, 3
    {  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x80c,   0000 1000   O   {  3, 11, 11, 2,  2, 3
    {  0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x905,   0000 1001   O   {  2, 0,  0, 8,  8, 11, 11, 2
    {  1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xa0f,   0000 1010   X
    {  1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0xb06,   0000 1011   O   {  2, 1,  1, 9,  9, 8,  8, 11, 11, 2
    {  3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xc0a,   0000 1100   O   {  1, 3,  3, 11, 11, 10, 10, 1
    {  0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1 }, // 0xd03,   0000 1101   O   {  1, 0,  0, 8,  8, 11, 11, 10, 10, 1
    {  3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1 }, // 0xe09,   0000 1110   O   {  0, 3,  3, 11, 11, 10, 10, 9,  9, 0
    {  9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xf00,   0000 1111   O   {  9, 8,  8, 11, 11, 10, 10, 9

    {  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x190,   0001 0000   O   {  4, 7,  7, 8,  8, 4
    {  4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x099,   0001 0001   O   {  3, 0,  0, 4,  4, 7,  7, 3
    {  0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x393,   0001 0010   X
    {  4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1 }, // 0x29a,   0001 0011   O   {  3, 1,  1, 9,  9, 4,  4, 7,  7, 3
    {  1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x596,   0001 0100   X
    {  3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1 }, // 0x49f,   0001 0101   X
    {  9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0x795,   0001 0110   X
    {  2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1 }, // 0x69c,   0001 0111   O   {  3, 2,  2, 10, 10, 9,  9, 4,  4, 7,  7, 3
    {  8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x99c,   0001 1000   X
    { 11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1 }, // 0x895,   0001 1001   O   {  2, 0,  0, 4,  4, 7,  7, 11, 11, 2
    {  9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0xb9f,   0001 1010   X
    {  4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1 }, // 0xa96,   0001 1011   O   {  2, 1,  1, 9,  9, 4,  4, 7,  7, 11, 11, 2
    {  3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1 }, // 0xd9a,   0001 1100   X
    {  1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1 }, // 0xc93,   0001 1101   O   {  1, 0,  0, 4,  4, 7,  7, 11, 11, 10, 10, 1
    {  4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1 }, // 0xf99,   0001 1110   X
    {  4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1 }, // 0xe90,   0001 1111   O   {  4, 7,  7, 11, 11, 10, 10, 9,  9, 4,

    {  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x230,   0010 0000   O   {  9, 5,  5, 4,  4, 9
    {  9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x339,   0010 0001   X
    {  0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x033,   0010 0010   O   {  0, 1,  1, 5,  5, 4,  4, 0
    {  8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0x13a,   0010 0011   O   {  3, 1,  1, 5,  5, 4,  4, 8,  8, 3
    {  1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x636,   0010 0100   X
    {  3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0x73f,   0010 0101   X
    {  5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1 }, // 0x435,   0010 0110   O   {  0, 2,  2, 10, 10, 5,  5, 4,  4, 0
    {  2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1 }, // 0x53c,   0010 0111   O   {  3, 2,  2, 10, 10, 5,  5, 4,  4, 8,  8, 3
    {  9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xa3c,   0010 1000   X
    {  0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0xb35,   0010 1001   X
    {  0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0x83f,   0010 1010   X
    {  2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1 }, // 0x936,   0010 1011   O   {  2, 1,  1, 5,  5, 4,  4, 8,  8, 11, 11, 2
    { 10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1 }, // 0xe3a,   0010 1100   X
    {  4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1 }, // 0xf33,   0010 1101   X
    {  5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1 }, // 0xc39,   0010 1110   O   {  0, 3,  3, 11, 11, 10, 10, 5,  5, 4,  4, 0
    {  5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0xd30,   0010 1111   O   {  5, 4,  4, 8,  8, 11, 11, 10, 10, 5

    {  9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x3a0,   0011 0000   O   {  5, 7,  7, 8,  8, 9,  9, 5
    {  9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1 }, // 0x2a9,   0011 0001   O   {  3, 0,  0, 9,  9, 5,  5, 7,  7, 3
    {  0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0x1a3,   0011 0010   O   {  0, 1,  1, 5,  5, 7,  7, 8,  8, 0
    {  1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x0aa,   0011 0011   O   {  3, 1,  1, 5,  5, 7,  7, 3
    {  9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1 }, // 0x7a6,   0011 0100   X
    { 10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1 }, // 0x6af,   0011 0101   X
    {  8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1 }, // 0x5a5,   0011 0110   O   {  0, 2,  2, 10, 10, 5,  5, 7,  7, 8,  8, 0
    {  2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0x4ac,   0011 0111   O   {  3, 2,  2, 10, 10, 5,  5, 7,  7, 3
    {  7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1 }, // 0xbac,   0011 1000   X
    {  9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1 }, // 0xaa5,   0011 1001   O   {  2, 0,  0, 9,  9, 5,  5, 7,  7, 11, 11, 2
    {  2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1 }, // 0x9af,   0011 1010   X
    { 11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0x8a6,   0011 1011   O   {  2, 1,  1, 5,  5, 7,  7, 11, 11, 2
    {  9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1 }, // 0xfaa,   0011 1100   X
    {  5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1 }, // 0xea3,   0011 1101   X
    { 11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1 }, // 0xda9,   0011 1110   X
    { 11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xca0,   0011 1111   O   {  5, 7,  7, 11, 11, 10, 10, 5

    { 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x460,   0100 0000   O   { 10, 6,  6, 5,  5, 10
    {  0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x569,   0100 0001   X
    {  9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x663,   0100 0010   X
    {  1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0x76a,   0100 0011   X
    {  1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x066,   0100 0100   O   {  1, 2,  2, 6,  6, 5,  5, 1
    {  1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1 }, // 0x16f,   0100 0101   X
    {  9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0x265,   0100 0110   O   {  0, 2,  2, 6,  6, 5,  5, 9,  9, 0
    {  5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1 }, // 0x36c,   0100 0111   O   {  3, 2,  2, 6,  6, 5,  5, 9,  9, 8,  8, 3
    {  2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xc6c,   0100 1000   X
    { 11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0xd65,   0100 1001   X
    {  0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0xe6f,   0100 1010   X
    {  5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1 }, // 0xf66,   0100 1011   X
    {  6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1 }, // 0x86a,   0100 1100   O   {  5, 1,  1, 3,  3, 11, 11, 6,  6, 5
    {  0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1 }, // 0x963,   0100 1101   O   {  1, 0,  0, 8,  8, 11, 11, 6,  6, 5,  5, 1
    {  3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1 }, // 0xa69,   0100 1110   O   {  0, 3,  3, 11, 11, 6,  6, 5,  5, 9,  9, 0
    {  6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1 }, // 0xb60,   0100 1111   O   {  6, 5,  5, 9,  9, 8,  8, 11, 11, 6

    {  5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x5f0,   0101 0000   X
    {  4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1 }, // 0x4f9,   0101 0001   X
    {  1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0x7f3,   0101 0010   X
    { 10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1 }, // 0x6fa,   0101 0011   X
    {  6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1 }, // 0x1f6,   0101 0100   X
    {  1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1 }, // 0x0ff,   0101 0101   X
    {  8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1 }, // 0x3f5,   0101 0110   X
    {  7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1 }, // 0xxfc,   0101 0111   X
    {  3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0xdfc,   0101 1000   X
    {  5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1 }, // 0xcf5,   0101 1001   X
    {  0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1 }, // 0xfff,   0101 1010   X
    {  9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1 }, // 0xef6,   0101 1011   X
    {  8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1 }, // 0x9fa,   0101 1100   X
    {  5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1 }, // 0x8f3,   0101 1101   X
    {  0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1 }, // 0xbf9,   0101 1110   X
    {  6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1 }, // 0xaf0,   0101 1111   O   {  4, 7,  7, 11, 11, 6,  6, 5,  5, 9,  9, 4

    { 10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x650,   0110 0000   O   {  6, 4,  4, 9,  9, 10, 10, 6
    {  4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1 }, // 0x759,   0110 0001   X
    { 10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1 }, // 0x453,   0110 0010   O   {  0, 1,  1, 10, 10, 6,  6, 4,  4, 0
    {  8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1 }, // 0x55a,   0110 0011   O   {  3, 1,  1, 10, 10, 6,  6, 4,  4, 8,  8, 3
    {  1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1 }, // 0x256,   0110 0100   O   {  1, 2,  2, 6,  6, 4,  4, 9,  9, 1
    {  3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1 }, // 0x35f,   0110 0101   X
    {  0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x055,   0110 0110   O   {  0, 2,  2, 6,  6, 4,  4, 0
    {  8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0x15c,   0110 0111   O   {  3, 2,  2, 6,  6, 4,  4, 8,  8, 3
    { 10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1 }, // 0xe5c,   0110 1000   X
    {  0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1 }, // 0xf55,   0110 1001   X
    {  3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1 }, // 0xc5f,   0110 1010   X
    {  6 , 4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1 }, // 0xd56,   0110 1011   O   {  2, 1,  1, 10, 10, 6,  6, 4,  4, 8,  8, 11, 11, 2
    {  9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1 }, // 0xa5a,   0110 1100   O   {  1, 3,  3, 11, 11, 6,  6, 4,  4, 9,  9, 1
    {  8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1 }, // 0xb53,   0110 1101   X
    {  3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1 }, // 0x859,   0110 1110   O   {  0, 3,  3, 11, 11, 6,  6, 4,  4, 0
    {  6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x950,   0110 1111   O   {  6, 4,  4, 8,  8, 11, 11, 6

    {  7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1 }, // 0x7c0,   0111 0000   O   {  6, 7,  7, 8,  8, 9,  9, 10, 10, 6
    {  0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1 }, // 0x6c9,   0111 0001   O   {  0, 10, 10, 6,  6, 7,  7, 3,  3, 0
    { 10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1 }, // 0x5c3,   0111 0010   O   {  0, 1,  1, 10, 10, 6,  6, 7,  7, 8,  8, 0
    { 10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1 }, // 0x4ca,   0111 0011   O   {  3, 1,  1, 10, 10, 6,  6, 7,  7, 3
    {  1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1 }, // 0x3c6,   0111 0100   O   {  1, 2,  2, 6,  6, 7,  7, 8,  8, 9,  9, 1
    {  2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1 }, // 0x2cf,   0111 0101   X
    {  7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1 }, // 0x1c5,   0111 0110   O   {  0, 2,  2, 6,  6, 7,  7, 8,  8, 0
    {  7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x0cc,   0111 0111   O   {  3, 2,  2, 6,  6, 7,  7, 3
    {  2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1 }, // 0xfcc,   0111 1000   X
    {  2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1 }, // 0xec5,   0111 1001   X
    {  1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1 }, // 0xdcf,   0111 1010   X
    { 11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1 }, // 0xcc6,   0111 1011   O   {  2, 1,  1, 10, 10, 6,  6, 7,  7, 11, 11, 2
    {  8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3 , 6, -1 }, // 0xbca,   0111 1100   X
    {  0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xac3,   0111 1101   X
    {  7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1 }, // 0x9c9,   0111 1110   O   {  0, 3,  3, 11, 11, 6,  6, 7,  7, 8,  8, 0
    {  7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x8c0,   0111 1111   O   {  6, 7,  7, 11, 11, 6

    {  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x8c0,   1000 0000   O   {  7, 6,  6, 11, 11, 7
    {  3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x9c9,   1000 0001   X
    {  0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xac3,   1000 0010   X
    {  8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0xbca,   1000 0011   X
    { 10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xcc6,   1000 0100   X
    {  1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0xdcf,   1000 0101   X
    {  2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0xec5,   1000 0110   X
    {  6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1 }, // 0xfcc,   1000 0111   X
    {  7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x0cc,   1000 1000   O   {  2, 3,  3, 7,  7, 6,  6, 2
    {  7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1 }, // 0x1c5,   1000 1001   O   {  2, 0,  0, 8,  8, 7,  7, 6,  6, 2
    {  2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1 }, // 0x2cf,   1000 1010   X
    {  1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1 }, // 0x3c6,   1000 1011   O   {  2, 1,  1, 9,  9, 8,  8, 7,  7, 6,  6, 2
    { 10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0x4ca,   1000 1100   O   {  1, 3,  3, 7,  7, 6,  6, 10, 10, 1
    { 10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1 }, // 0x5c3,   1000 1101   O   {  1, 0,  0, 8,  8, 7,  7, 6,  6, 10, 10, 1
    {  0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1 }, // 0x6c9,   1000 1110   O   {  0, 3,  3, 7,  7, 6,  6, 10, 10, 9,  9, 0
    {  7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1 }, // 0x7c0,   1000 1111   O   {  7, 6,  6, 10, 10, 9,  9, 8,  8, 7

    {  6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x950,   1001 0000   O   {  4, 6,  6, 11, 11, 8,  8, 4
    {  3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0x859,   1001 0001   O   {  3, 0,  0, 4,  4, 6,  6, 11, 11, 3
    {  8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1 }, // 0xb53,   1001 0010   X
    {  9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1 }, // 0xa5a,   1001 0011   O   {  1, 9,  9, 4,  4, 6,  6, 11, 11, 3,  3, 1
    {  6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1 }, // 0xd56,   1001 0100   X
    {  1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1 }, // 0xc5f,   1001 0101   X
    {  4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1 }, // 0xf55,   1001 0110   X
    { 10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1 }, // 0xe5c,   1001 0111   X
    {  8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1 }, // 0x15c,   1001 1000   O   {  2, 3,  3, 8,  8, 4,  4, 6,  6, 2
    {  0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x055,   1001 1001   O   {  2, 0,  0, 4,  4, 6,  6, 2
    {  1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1 }, // 0x35f,   1001 1010   X
    {  1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0x256,   1001 1011   O   {  2, 1,  1, 9,  9, 4,  4, 6,  6, 2
    {  8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1 }, // 0x55a,   1001 1100   O   {  1, 3,  3, 8,  8, 4,  4, 6,  6, 10, 10, 1
    { 10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1 }, // 0x453,   1001 1101   O   {  1, 0,  0, 4,  4, 6,  6, 10, 10, 1
    {  4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1 }, // 0x759,   1001 1110   X
    { 10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x650,   1001 1111   O   {  4, 6,  6, 10, 10, 9,  9, 4

    {  4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xaf0,   1010 0000   X
    {  0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1 }, // 0xbf9,   1010 0001   X
    {  5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0x8f3,   1010 0010   X
    { 11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1 }, // 0x9fa,   1010 0011   X
    {  9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0xef6,   1010 0100   X
    {  6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1 }, // 0xfff,   1010 0101   X
    {  7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1 }, // 0xcf5,   1010 0110   X
    {  3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1 }, // 0xdfc,   1010 0111   X
    {  7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1 }, // 0x2fc,   1010 1000   X
    {  9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1 }, // 0x3f5,   1010 1001   X
    {  3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1 }, // 0x0ff,   1010 1010   X
    {  6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1 }, // 0x1f6,   1010 1011   X
    {  9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1 }, // 0x6fa,   1010 1100   X
    {  1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1 }, // 0x7f3,   1010 1101   X
    {  4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1 }, // 0x4f9,   1010 1110   X
    {  7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1 }, // 0x5f0,   1010 1111   O   {  5, 4,  4, 8,  8, 7,  7, 6,  6, 10, 10, 5

    {  6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1 }, // 0xb60,   1011 0000   O   {  5, 6,  6, 11, 11, 8,  8, 9,  9, 5
    {  3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1 }, // 0xa69,   1011 0001   O   {  5, 6,  6, 11, 11, 3,  3, 0,  0, 9,  9, 5
    {  0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1 }, // 0x963,   1011 0010   O   {  0, 1,  1, 5,  5, 6,  6, 11, 11, 8,  8, 0
    {  6, 11,  3,  6,  3,  5,  5,  3 , 1, -1, -1, -1, -1, -1, -1, -1 }, // 0x86a,   1011 0011   O   {  3, 1,  1, 5,  5, 6,  6, 11, 11, 3
    {  1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1 }, // 0xf66,   1011 0100   X
    {  0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1 }, // 0xe6f,   1011 0101   X
    { 11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1 }, // 0xd65,   1011 0110   X
    {  3,  6, 11,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1 }, //{  2, 11,  3,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xc6c,  1011 0111   X
    {  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1 }, // 0x36c,   1011 1000   O   {  2, 3,  3, 8,  8, 9,  9, 5,  5, 6,  6, 2
    {  9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1 }, // 0x265,   1011 1001   O   {  2, 0,  0, 9,  9, 5,  5, 6,  6, 2
    {  1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1 }, // 0x16f,   1011 1010   X
    {  1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x066,   1011 1011   O   {  2, 1,  1, 5,  5, 6,  6, 2
    {  1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1 }, // 0x76a,   1011 1100   X
    { 10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6, 0 , -1, -1, -1, -1 }, // 0x663,   1011 1101   O   {  1, 0,  0, 9,  9, 5,  5, 6,  6, 10, 10, 1
    {  0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x569,   1011 1110   X
    { 10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x460,   1011 1111   O   { 10, 5,  5, 6,  6, 10

    { 11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xca0,   1100 0000   O   {  7, 5,  5, 10, 10, 11, 11, 7
    { 11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1 }, // 0xda9,   1100 0001   X
    {  5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1 }, // 0xea3,   1100 0010   X
    { 10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1 }, // 0xfaa,   1100 0011   X
    { 11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1 }, // 0x8a6,   1100 0100   O   {  1, 2,  2, 11, 11, 7,  7, 5,  5, 1
    {  0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1 }, // 0x9af,   1100 0101   X
    {  9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1 }, // 0xaa5,   1100 0110   O   {  0, 2,  2, 11, 11, 7,  7, 5,  5, 9,  9, 0
    {  7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1 }, // 0xbac,   1100 0111   X
    {  2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0x4ac,   1100 1000   O   {  2, 3,  3, 7,  7, 5,  5, 10, 10, 2
    {  8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1 }, // 0x5a5,   1100 1001   O   {  2, 0,  0, 8,  8, 7,  7, 5,  5, 10, 10, 2
    {  9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1 }, // 0x6af,   1100 1010   X
    {  9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1 }, // 0x7a6,   1100 1011   X
    {  1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x0aa,   1100 1100   O   {  1, 3,  3, 7,  7, 5,  5, 1
    {  0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1 }, // 0x1a3,   1100 1101   O   {  1, 0,  0, 8,  8, 7,  7, 5,  5, 1
    {  9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1 }, // 0x2a9,   1100 1110   O   {  0, 3,  3, 7,  7, 5,  5, 9,  9, 0
    {  9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x3a0,   1100 1111   O   {  7, 5,  5, 9,  9, 8,  8, 7

    {  5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1 }, // 0xd30,   1101 0000   O   {  4, 5,  5, 10, 10, 11, 11, 8,  8, 4
    {  5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1 }, // 0xc39,   1101 0001   O   {  3, 0,  0, 4,  4, 5,  5, 10, 10, 11, 11, 3,
    {  0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1 }, // 0xf33,   1101 0010   X
    { 10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1 }, // 0xe3a,   1101 0011   X
    {  2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1 }, // 0x936,   1101 0100   O   {  1, 2,  2, 11, 11, 8,  8, 4,  4, 5,  5, 1
    {  0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1 }, // 0x83f,   1101 0101   X
    {  0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1 }, // 0xb35,   1101 0110   X
    {  9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xa3c,   1101 0111   X
    {  2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1 }, // 0x53c,   1101 1000   O   {  2, 3,  3, 8,  8, 4,  4, 5,  5, 10, 10, 2
    {  5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1 }, // 0x435,   1101 1001   O   {  2, 0,  0, 4,  4, 5,  5, 10, 10, 2
    {  3, 10,  2,  3,  5, 10,  3,  8,  5 , 4,  5,  8,  0,  1,  9, -1 }, // 0x73f,   1101 1010   X
    {  5, 10,  2,  5,  2 , 4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1 }, // 0x636,   1101 1011   O   {  1, 9,  9, 4,  4, 5,  5, 10, 10, 2,  2, 1
    {  8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1 }, // 0x13a,   1101 1100   O   {  1, 3,  3, 8,  8, 4,  4, 5,  5, 1
    {  0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x033,   1101 1101   O   {  1, 0,  0, 4,  4, 5,  5, 1
    {  8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3 , 5, -1, -1, -1, -1 }, // 0x339,   1101 1110   O   {  0, 3,  3, 8,  8, 4,  4, 5,  5, 9,  9, 0
    {  9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x230,   1101 1111   O   {  9, 4,  4, 5,  5, 9

    {  4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0xe90,   1110 0000   O   {  7, 4,  4, 9,  9, 10, 10, 11, 11, 7
    {  0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1 }, // 0xf99,   1110 0001   X
    {  1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1 }, // 0xc93,   1110 0010   O   {  0, 1,  1, 10, 10, 11, 11, 7,  7, 4,  4, 0
    {  3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1 }, // 0xd9a,   1110 0011   X
    {  4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1 }, // 0xa96,   1110 0100   O   {  1, 2,  2, 11, 11, 7,  7, 4,  4, 9,  9, 1
    {  9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1 }, // 0xb9f,   1110 0101   X
    { 11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1 }, // 0x895,   1110 0110   O   {  0, 2,  2, 11, 11, 7,  7, 4,  4, 0
    { 11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1 }, // 0x99c,   1110 0111   O   {  3, 2,  2, 11, 11, 7,  7, 4,  4, 8,  8, 3
    {  2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1 }, // 0x69c,   1110 1000   O   {  2, 3,  3, 7,  7, 4,  4, 9,  9, 10, 10, 2
    {  9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1 }, // 0x795,   1110 1001   X
    {  3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1 }, // 0x49f,   1110 1010   X
    {  1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x596,   1110 1011   X
    {  4,  9,  1 , 4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1 }, // 0x29a,   1110 1100   O   {  1, 3,  3, 7,  7, 4,  4, 9,  9, 1
    {  4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1 }, // 0x393,   1110 1101   O   {  1, 0,  0, 8,  8, 7,  7, 4,  4, 9,  9, 1
    {  4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x099,   1110 1110   O   {  0, 3,  3, 7,  7, 4,  4, 0
    {  4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x190,   1110 1111   O   {  4, 8,  8, 7,  7, 4

    {  9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xf00,   1111 0000   O   {  8, 9,  9, 10, 10, 11, 11, 8
    {  3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1 }, // 0xe09,   1111 0001   O   {  3, 0,  0, 9,  9, 10, 10, 11, 11, 3
    {  0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1 }, // 0xd03,   1111 0010   O   {  0, 1,  1, 10, 10, 11, 11, 8,  8, 0
    {  3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0xc0a,   1111 0011   O   {  3, 1,  1, 10, 10, 11, 11, 3
    {  1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1 }, // 0xb06,   1111 0100   O   {  1, 2,  2, 11, 11, 8,  8, 9,  9, 1
    {  3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1 }, // 0xa0f,   1111 0101   O   {  0, 9,  9, 1,  1, 2,  2, 11, 11, 3,  3, 0
    {  0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x905,   1111 0110   O   {  0, 2,  2, 11, 11, 8,  8, 0
    {  3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x80c,   1111 0111   O   {  3, 2,  2, 11, 11, 3
    {  2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1 }, // 0x70c,   1111 1000   O   {  2, 3,  3, 8,  8, 9,  9, 10, 10, 2
    {  9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x605,   1111 1001   O   {  2, 0,  0, 9,  9, 10, 10, 2
    {  2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1 }, // 0x50f,   1111 1010   O   {  0, 1,  1, 10, 10, 2,  2, 3,  3, 8,  8, 0
    {  1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x406,   1111 1011   O   {  1, 10, 10, 2,  2, 1
    {  1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x30a,   1111 1100   O   {  1, 3,  3, 8,  8, 9,  9, 1
    {  0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x203,   1111 1101   O   {  0, 9,  9, 1,  1, 0
    {  0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, // 0x109,   1111 1110   O   {  0, 3,  3, 8,  8, 0
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }  // 0x000,   1111 1111   X
};

const int MarchingCubesSolver::_squareEdgeFlags[16] = {
    0x000, 0x009, 0x003, 0x00a, 0x006, 0x00f, 0x005, 0x00c,
    0x00c, 0x005, 0x00f, 0x006, 0x00a, 0x003, 0x009, 0x000
};

const int MarchingCubesSolver::_edgeConnection2D[4][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}
};

const int MarchingCubesSolver::_triangleConnectionTable2D[16][13] = {
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  7,  2,  6,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  1,  6,  1,  2,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  7,  7,  1,  6,  1,  2,  6, -1, -1, -1, -1 },
    {  7,  6,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  6,  0,  6,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  7,  6,  6,  7,  4,  6,  4,  5,  1,  5,  4, -1 },
    {  0,  6,  3,  0,  5,  6,  0,  1,  5, -1, -1, -1, -1 },
    {  7,  5,  3,  5,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  4,  3,  4,  5,  3,  5,  2, -1, -1, -1, -1 },
    {  2,  3,  7,  2,  7,  4,  2,  4,  1, -1, -1, -1, -1 },
    {  0,  1,  3,  1,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
};