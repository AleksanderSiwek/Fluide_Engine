#include "cuda_array_utils.hpp"

#include "parallel_utils.hpp"
#include <iostream>
#include <stdio.h>

CUDA_Vector3 Vector3ToCUDA_Vector3(const Vector3<double>& vect)
{
    CUDA_Vector3 cudaVector = {0, 0, 0};
    cudaVector.x = vect.x;
    cudaVector.y = vect.y;
    cudaVector.z = vect.z;
    return cudaVector;
}

CUDA_Int3 Vector3SizeToCUDA_Int3(const Vector3<size_t>& vect)
{
    CUDA_Int3 intSize = {0, 0, 0};
    intSize.x = static_cast<int>(vect.x);
    intSize.y = static_cast<int>(vect.y);
    intSize.z = static_cast<int>(vect.z);
    return intSize; 
}

Vector3<double> CUDA_Vector3ToVector3(const CUDA_Vector3& vect)
{
    return Vector3<double>(vect.x, vect.y, vect.z);
}

Vector3<size_t> CUDA_Int3ToVector3Size(const CUDA_Int3& vect)
{
        return Vector3<size_t>((size_t)vect.x, (size_t)vect.y, (size_t)vect.z);
}

__device__ double CUDA_Vector3Dot(CUDA_Vector3 a, CUDA_Vector3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ CUDA_Vector3 CUDA_Vector3Project(CUDA_Vector3 a, CUDA_Vector3 normal)
{
    CUDA_Vector3 projected = a;
    double dot = CUDA_Vector3Dot(a, normal);
    projected.x = a.x - (normal.x * dot);
    projected.y = a.y - (normal.y * dot);
    projected.z = a.z - (normal.z * dot);
    return projected;
}

__device__ double CUDA_Vector3GetLength(CUDA_Vector3 vect)
{
    return sqrt(max(vect.x * vect.x + vect.y * vect.y + vect.z * vect.z, 0.0));
}

__device__ CUDA_Vector3 CUDA_Vector3GetNormalised(CUDA_Vector3 vect)
{
    CUDA_Vector3 normalisedVect = {0, 0, 0};
    double length = CUDA_Vector3GetLength(vect);
    normalisedVect.x = vect.x / length;
    normalisedVect.y = vect.y / length;
    normalisedVect.z = vect.z / length;
    return normalisedVect;
}

__device__ CUDA_Vector3 CUDA_GridIdxToPosition(CUDA_Int3 idx, CUDA_Vector3 gridSpacing, CUDA_Vector3 dataOrigin)
{
    CUDA_Vector3 position = {0, 0, 0};
    position.x = dataOrigin.x + gridSpacing.x * (double)idx.x;
    position.y = dataOrigin.y + gridSpacing.y * (double)idx.y;
    position.z = dataOrigin.z + gridSpacing.z * (double)idx.z;
    return position;
}

__device__ double CUDA_Lerp(const double a, const double b, double factor)
{
    return (1 - factor) * a + factor * b;
}

__device__ double CUDA_Bilerp(const double x00, const double x10, const double x01, const double x11, double factorX, double factorY)     
{
    return CUDA_Lerp(CUDA_Lerp(x00, x10, factorX), CUDA_Lerp(x01, x11, factorX), factorY);
}  

__device__ double CUDA_Trilerp(const double x000, const double x100, const double x010, const double x110, const double x001, 
                          const double x101, const double x011, const double x111, double factorX, double factorY, double factorZ)     
{
    return CUDA_Lerp(CUDA_Bilerp(x000, x100, x010, x110, factorX, factorY), CUDA_Bilerp(x001, x101, x011, x111, factorX, factorY), factorZ);
}  


__device__ void  CUDA_GetBarycentric(double x, int iLow, int iHigh, int* i, double* f) 
{
    double s = floor(x);
    *i = (int)s;
    int siLow = (int)iLow;
    int siHigh = (int)iHigh;

    int offset = -siLow;
    siLow += offset;
    siHigh += offset;

    if (siLow == siHigh) {
        *i = siLow;
        *f = 0;
    } else if (*i < siLow) {
        *i = siLow;
        *f = 0;
    } else if (*i > siHigh - 1) {
        *i = siHigh - 1;
        *f = 1;
    } else {
        *f = (double)(x - s);
    }

    *i -= offset;
}

__device__ double CUDA_SampleArray3(double* array, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, CUDA_Int3 size, CUDA_Vector3 position)
{
    int i, j, k;
    double factorX, factorY, factorZ;
    i = j = k = 0;
    factorX = factorY = factorZ = 0;

    CUDA_Vector3 normalizedPoistion = {0, 0, 0}; 
    normalizedPoistion.x = (position.x - origin.x) / gridSpacing.x;
    normalizedPoistion.y = (position.y - origin.y) / gridSpacing.y;
    normalizedPoistion.z = (position.z - origin.z) / gridSpacing.z;

    int sizeX = size.x;
    int sizeY = size.y;
    int sizeZ = size.z;

    CUDA_GetBarycentric(normalizedPoistion.x, 0, sizeX - 1, &i, &factorX);
    CUDA_GetBarycentric(normalizedPoistion.y, 0, sizeY - 1, &j, &factorY);
    CUDA_GetBarycentric(normalizedPoistion.z, 0, sizeZ - 1, &k, &factorZ);

    const size_t ip1 = i + 1 < sizeX - 1 ? i + 1 : sizeX - 1;
    const size_t jp1 = j + 1 < sizeY - 1 ? j + 1 : sizeY - 1;
    const size_t kp1 = k + 1 < sizeZ - 1 ? k + 1 : sizeZ - 1;

    return CUDA_Trilerp(array[i + sizeX * (j + sizeY * k)],
                        array[ip1 + sizeX * (j + sizeY * k)],
                        array[i + sizeX * (jp1 + sizeY * k)],
                        array[ip1 + sizeX * (jp1 + sizeY * k)],
                        array[i + sizeX * (j + sizeY * kp1)],
                        array[ip1 + sizeX * (j + sizeY * kp1)],
                        array[i + sizeX * (jp1 + sizeY * kp1)],
                        array[ip1 + sizeX * (jp1 + sizeY * kp1)],
                        factorX,
                        factorY,
                        factorZ);
}

__device__ CUDA_Vector3 CUDA_SampleFaceCenteredGrid3(double* dataX, double* dataY, double* dataZ, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, CUDA_Int3 size, CUDA_Vector3 position)
{
    CUDA_Vector3 sampledVetor = {0, 0, 0};
    sampledVetor.x = CUDA_SampleArray3(dataX, origin, gridSpacing, size, position);
    sampledVetor.y = CUDA_SampleArray3(dataY, origin, gridSpacing, size, position);
    sampledVetor.z = CUDA_SampleArray3(dataZ, origin, gridSpacing, size, position);
    return sampledVetor;
}

__device__ CUDA_Vector3 CUDA_GradientArray3(double* array, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, CUDA_Int3 size, CUDA_Vector3 position)
{
    CUDA_Int3* indexes = (CUDA_Int3*)malloc(8 * sizeof(CUDA_Int3));
    double* weights = (double*)malloc(8 * sizeof(double));
    CUDA_GetCooridnatesAndWeights(size, origin, gridSpacing, position, indexes, weights);
    // const auto& ds = GetSize();

    // Vector3<double> result;
    CUDA_Vector3 result = {0, 0, 0};
    for(int idx = 0; idx < 8; idx++)
    {
        int i = indexes[idx].x;
        int j = indexes[idx].y;
        int k = indexes[idx].z;
        double left = array[(i > 0) ? i - 1 : i + size.x * (j + size.y * k)];
        double right = array[(i + 1 < size.x) ? i + 1 : i + size.x * (j + size.y * k)];
        double down = array[i + size.x * ((j > 0) ? j - 1 : j + size.y * k)];
        double up = array[i + size.x * ((j + 1 < size.y) ? j + 1 : j + size.y * k)];
        double back = array[i + size.x * (j + size.y * (k > 0) ? k - 1 : k)];
        double front = array[i + size.x * (j + size.y * (k + 1 < size.z) ? k + 1 : k)];
        CUDA_Vector3 tmpVect = {right - left, up - down, front - back};
        result.x += weights[idx] * 0.5 * tmpVect.x / gridSpacing.x;
        result.y += weights[idx] * 0.5 * tmpVect.y / gridSpacing.y;
        result.z += weights[idx] * 0.5 * tmpVect.z / gridSpacing.z;
    }

    free(indexes);
    free(weights);
    return result;
}


__device__ void CUDA_GetCooridnatesAndWeights(CUDA_Int3 size, CUDA_Vector3 origin, CUDA_Vector3 gridSpacing, 
                              CUDA_Vector3 x, CUDA_Int3* indexes, double* weights)
{
    int i = 0, j = 0, k = 0;
    double fx = 0, fy = 0, fz = 0;

    const int iSize = size.x;
    const int jSize = size.y;
    const int kSize = size.z;

    CUDA_Vector3 normalizedX = x; 
    normalizedX.x = (x.x - origin.x) / gridSpacing.x;
    normalizedX.y = (x.y - origin.y) / gridSpacing.y;
    normalizedX.z = (x.z - origin.z) / gridSpacing.z;

    CUDA_GetBarycentric(normalizedX.x, 0, iSize - 1, &i, &fx);
    CUDA_GetBarycentric(normalizedX.y, 0, jSize - 1, &j, &fy);
    CUDA_GetBarycentric(normalizedX.z, 0, kSize - 1, &k, &fz);

    const int ip1 = i + 1 < iSize - 1 ? i + 1 : iSize - 1;
    const int jp1 = j + 1 < jSize - 1 ? j + 1 : jSize - 1;
    const int kp1 = k + 1 < kSize - 1 ? k + 1 : kSize - 1;

    indexes[0] = {i, j, k};
    indexes[1] = {ip1, j, k};
    indexes[2] = {i, jp1, k};
    indexes[3] = {ip1, jp1, k};
    indexes[4] = {i, j, kp1};
    indexes[5] = {ip1, j, kp1};
    indexes[6] = {i, jp1, kp1};
    indexes[7] = {ip1, jp1, kp1};

    weights[0] = (1.00 - fx) * (1.00 - fy) * (1.00 - fz);
    weights[1] = fx * (1.00 - fy) * (1.00 - fz);
    weights[2] = (1.00 - fx) * fy * (1.00 - fz);
    weights[3] = fx * fy * (1.00 - fz);
    weights[4] = (1.00 - fx) * (1.00 - fy) * fz;
    weights[5] = fx * (1.00 - fy) * fz;
    weights[6] = (1.00 - fx) * fy * fz; 
    weights[7] = fx * fy * fz; 
}

__global__ void CUDA_ExtrapolateToRegion(double* input, int* valid, size_t numberOfIterations, double* output, const size_t sizeX, const size_t sizeY, const size_t sizeZ)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + sizeX * (j + sizeY * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < sizeX && j < sizeY && k < sizeZ)
    { 
        output[idx] = input[idx];

        double tmp = 0;
        int valid1 = 0;
        for(unsigned int iter = 0; iter < numberOfIterations; ++iter)
        {
            double sum = 0;
            unsigned int count = 0;

            if (!valid[idx]) 
            {
                if (i + 1 < sizeX && valid[(i + 1) + sizeX * (j + sizeY * k)]) 
                {
                    sum += output[(i + 1) + sizeX * (j + sizeY * k)];
                    ++count;
                }
                if (i > 0 && valid[(i - 1) + sizeX * (j + sizeY * k)]) 
                {
                    sum += output[(i - 1) + sizeX * (j + sizeY * k)];
                    ++count;
                }

                if (j + 1 < sizeY && valid[i + sizeX * (j + sizeY * k)]) 
                {
                    sum += output[i + sizeX * (j + sizeY * k)];
                    ++count;
                }
                if (j > 0 && valid[i + sizeX * ((j - 1) + sizeY * k)]) 
                {
                    sum += output[i + sizeX * ((j - 1) + sizeY * k)];
                    ++count;
                }

                if (k + 1 < sizeZ && valid[i + sizeX * (j + sizeY * k)]) 
                {
                    sum += output[i + sizeX * (j + sizeY * k)];
                    ++count;
                }
                if (k > 0 && valid[i + sizeX * (j + sizeY * (k - 1))]) 
                {
                    sum += output[i + sizeX * (j + sizeY * (k - 1))];
                    ++count;
                }

                if (count > 0) 
                {
                    output[idx] = sum / count;
                    valid1 = 1;
                }
            } 
            else 
            {
                valid1 = 1;
            }

            __syncthreads();

            tmp = valid[idx];
            valid[idx] = valid1;
            valid1 = tmp;
        }
    }
}

__device__ double CUDA_Clamp(double val, double minVal, double maxVal)
{
    if(val < minVal)
    {
        return minVal;
    }
    else if(val > maxVal)
    {
        return maxVal;
    }
    else
    {
        return val;
    }
}

__global__ void CUDA_FillArray3(double* array, double val, CUDA_Int3 size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        array[idx] = val;
    }
}

__global__ void CUDA_CopyArray3(double* destination, double* source, CUDA_Int3 size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);
    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        destination[idx] = source[idx];
    }

}

void WrappedCuda_ExtrapolateToRegion(const Array3<double>& input, const Array3<int>& valid, size_t numberOfIterations, Array3<double>& output)
{
    const auto& size = input.GetSize();
    const unsigned int vectorSize = (unsigned int)size.x * (unsigned int)size.y * (unsigned int)size.z;
    int* h_markers;
    double* h_input;
    double* h_output;
    int* d_markers;
    double* d_input;
    double* d_output;

    h_markers = (int*)malloc(vectorSize * sizeof(int));
    h_input = (double*)malloc(vectorSize * sizeof(double));
    h_output = (double*)malloc(vectorSize * sizeof(double));

    // parallel_utils::ForEach3(size.x, size.y, size.z, [&](size_t i, size_t j, size_t k)
    // {
    //     h_input[i + size.x * (j + size.y * k)] = input(i, j, k);
    //     h_markers[i + size.x * (j + size.y * k)] = valid(i, j, k);
    // });

    cudaMalloc((void **)&d_markers, vectorSize * sizeof(int));
    cudaMalloc((void **)&d_input, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_output, vectorSize * sizeof(double));
    
    cudaMemcpy(d_input, &(input.GetRawData())[0], vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_markers, &(valid.GetRawData())[0], vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    int threadsInX = 4;
    int threadsInY = 4;
    int threadsInZ = 4;

    int blocksInX = (int)std::ceil(((double)size.x) / threadsInX);
    int blocksInY = (int)std::ceil(((double)size.y) / threadsInY);
    int blocksInZ = (int)std::ceil(((double)size.z) / threadsInZ);

    dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(threadsInX, threadsInY, threadsInZ);

    CUDA_ExtrapolateToRegion<<<dimGrid, dimBlock>>>(d_input, d_markers, numberOfIterations, d_output, size.x, size.y, size.z);

    cudaMemcpy(h_output, d_output, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);

    parallel_utils::ForEach3(size.x, size.y, size.z, [&](size_t i, size_t j, size_t k)
    {
        output(i, j, k) = h_output[i + size.x * (j + size.y * k)];
    });

    free(h_markers);
    free(h_input);
    free(h_output);
    cudaFree(d_markers);
    cudaFree(d_input);
    cudaFree(d_output);
}


