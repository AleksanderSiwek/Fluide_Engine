#ifndef _CUDA_ARRAY_UTILS_HPP
#define _CUDA_ARRAY_UTILS_HPP

#include <cuda_runtime.h>

#include "array3.hpp"


typedef struct
{
    double x;
    double y;
    double z;
} CUDA_Vector3;

typedef struct
{
    size_t x;
    size_t y;
    size_t z;
} CUDA_Size3;

typedef struct
{
    int x;
    int y;
    int z;
} CUDA_Int3;


CUDA_Vector3 Vector3ToCUDA_Vector3(const Vector3<double>& vect);

CUDA_Int3 Vector3SizeToCUDA_Int3(const Vector3<size_t>& vect);

Vector3<double> CUDA_Vector3ToVector3(const CUDA_Vector3& vect);

Vector3<size_t> CUDA_Int3ToVector3Size(const CUDA_Int3& vect);

__device__ double CUDA_Vector3Dot(CUDA_Vector3 a, CUDA_Vector3 b);

__device__ CUDA_Vector3 CUDA_Vector3Project(CUDA_Vector3 a, CUDA_Vector3 normal);

__device__ double CUDA_Vector3GetLength(CUDA_Vector3 vect);

__device__ CUDA_Vector3 CUDA_Vector3GetNormalised(CUDA_Vector3 vect);

void WrappedCuda_ExtrapolateToRegion(const Array3<double>& input, const Array3<int>& valid, size_t numberOfIterations, Array3<double>& output);

__device__ CUDA_Vector3 CUDA_GridIdxToPosition(CUDA_Int3 idx, CUDA_Vector3 gridSpacing, CUDA_Vector3 dataOrigin);

__device__ double CUDA_Lerp(const double a, const double b, double factor);

__device__ double CUDA_Bilerp(const double x00, const double x10, const double x01, const double x11, double factorX, double factorY);    

__device__ double CUDA_Trilerp(const double x000, const double x100, const double x010, const double x110, const double x001, 
                          const double x101, const double x011, const double x111, double factorX, double factorY, double factorZ);     

__device__ double CUDA_SampleArray3(double* array, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, CUDA_Int3 size, CUDA_Vector3 position);

__device__ CUDA_Vector3 CUDA_SampleFaceCenteredGrid3(double* dataX, double* dataY, double* dataZ, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, CUDA_Int3 size, CUDA_Vector3 position);

__device__ CUDA_Vector3 CUDA_GradientArray3(double* array, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, CUDA_Int3 size, CUDA_Vector3 position);

__device__ void CUDA_GetBarycentric(double x, int iLow, int iHigh, int* i, double* f);

__device__ void CUDA_GetCooridnatesAndWeights(CUDA_Int3 size, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, CUDA_Vector3 x, CUDA_Int3* indexes, double* weights);

__device__ double CUDA_Clamp(double val, double minVal, double maxVal);

__global__ void CUDA_ExtrapolateToRegion(double* input, int* valid, size_t numberOfIterations, double* output, const size_t sizeX, const size_t sizeY, const size_t sizeZ);

__global__ void CUDA_FillArray3(double* array, double val, CUDA_Int3 size);

__global__ void CUDA_CopyArray3(double* destination, double* source, CUDA_Int3 size);

#endif // _CUDA_ARRAY_UTILS_HPP