#ifndef _CUDA_COLLISIONS_HPP
#define _CUDA_COLLISIONS_HPP

#include <cuda_runtime.h>

#include "../common/cuda_array_utils.hpp"


namespace CUDA_COLLISIONS
{
    // __device__ double CUDA_DistanceToPoint(CUDA_Vector3 p1, CUDA_Vector3 p2);
    // __device__ CUDA_Vector3 CUDA_ClossestPointOnTriangle(CUDA_Vector3 point, CUDA_Vector3 triPoint1, CUDA_Vector3 triPoint2), CUDA_Vector3 triPoint3);
    // __device__ double CUDA_DistanceToTriangle(CUDA_Vector3 point, CUDA_Vector3 triPoint1, CUDA_Vector3 triPoint2), CUDA_Vector3 triPoint3);
    // __device__ double CUDA_ADet(CUDA_Vector3 p1, CUDA_Vector3 p2);
    
    // __global__ size_t CUDA_ClosestTriangleIdx(CUDA_Vector3 p1, CUDA_Vector3 p2);
    // __global__ bool CUDA_IsInsideTriangleMesh(CUDA_Vector3 p1, CUDA_Vector3 p2);

    // static double DistanceToPoint(Vector3<double> p1, Vector3<double> p2);
    // static Vector3<double> ClossestPointOnTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3);
    // static double DistanceToTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3);
    // static size_t ClosestTriangleIdx(Vector3<double> point, const TriangleMesh& mesh);
    // static bool IsInsideTriangleMesh(const TriangleMesh& mesh, const Vector3<double>& point);
    // static double ADet(const Vector3<double>& point1, const Vector3<double>& point2, const Vector3<double>& point3);
}

#endif // _CUDA_COLLISIONS_HPP