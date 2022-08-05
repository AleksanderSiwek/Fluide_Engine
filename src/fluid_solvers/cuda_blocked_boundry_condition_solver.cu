#include "cuda_blocked_boundry_condition_solver.hpp"
#include "../common/cuda_array_utils.hpp"
#include "../grid_systems/fluid_markers.hpp"
#include "../common/cuda_array_utils.hpp"

#include <iostream>

__device__ double CUDA_FractionInsideSdf(double phi0, double phi1)
{
    if(phi0 < 0 && phi1 < 0)
    {
        return 1;
    }
    else if(phi0 < 0 && phi1 >= 0)
    {
        return phi0 / (phi0 - phi1);
    }
    else if(phi0 >= 0 && phi1 < 0)
    {
        return phi1 / (phi1 - phi0);
    }
    else
    {
        return 0;
    }
}

__device__ CUDA_Vector3 CUDA_ApplyFriction(CUDA_Vector3 vel, CUDA_Vector3 normal, double frictionCoeddicient)
{
    CUDA_Vector3 velt = CUDA_Vector3Project(vel, normal);
    double veltLen = CUDA_Vector3GetLength(velt);
    if(veltLen * veltLen > 0.0)
    {
        double veln = max(-CUDA_Vector3Dot(vel, normal), 0.0);
        double scaler = max(1.0 - frictionCoeddicient * veln / veltLen, 0.0);
        velt.x = velt.x * scaler;
        velt.y = velt.y * scaler;
        velt.z = velt.z * scaler;
    }
    return velt;
}

__global__ void CUDA_BuildMarkers(double* arrayX, int* markersX, double* arrayY, int* markersY, double* arrayZ, int* markersZ, 
                                 double* colliderSdf, CUDA_Int3 size, CUDA_Vector3 gridSpacing, CUDA_Vector3 origin, 
                                 CUDA_Vector3 dataXOrigin, CUDA_Vector3 dataYOrigin, CUDA_Vector3 dataZOrigin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        CUDA_Int3 idx3 = {i, j, k};
        CUDA_Vector3 pos = CUDA_GridIdxToPosition(idx3, gridSpacing, dataXOrigin);
        CUDA_Vector3 pos0 = {pos.x - 0.5 * gridSpacing.x, pos.y, pos.z};
        CUDA_Vector3 pos1 = {pos.x + 0.5 * gridSpacing.x, pos.y, pos.z};
        double phi0 = CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos0);
        double phi1 = CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos1);
        double frac = CUDA_FractionInsideSdf(phi0, phi1);
        frac = 1 - CUDA_Clamp(frac, 0.0, 1.0);
        if(frac > 0)
        {
            markersX[idx] = 1;
        }
        else
        {
            arrayX[idx] = 0;
            markersX[idx] = 0;
        }

        pos = CUDA_GridIdxToPosition(idx3, gridSpacing, dataYOrigin);
        pos0 = {pos.x, pos.y - 0.5 * gridSpacing.y, pos.z};
        pos1 = {pos.x, pos.y + 0.5 * gridSpacing.y, pos.z};
        phi0 = CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos0);
        phi1 = CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos1);
        frac = CUDA_FractionInsideSdf(phi0, phi1);
        frac = 1 - CUDA_Clamp(frac, 0.0, 1.0);
        if(frac > 0)
        {
            markersY[idx] = 1;
        }
        else
        {
            arrayY[idx] = 0;
            markersY[idx] = 0;
        }

        pos = CUDA_GridIdxToPosition(idx3, gridSpacing, dataZOrigin);
        pos0 = {pos.x, pos.y, pos.z - 0.5 * gridSpacing.z};
        pos1 = {pos.x, pos.y, pos.z + 0.5 * gridSpacing.z};
        phi0 = CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos0);
        phi1 = CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos1);
        frac = CUDA_FractionInsideSdf(phi0, phi1);
        frac = 1 - CUDA_Clamp(frac, 0.0, 1.0);
        if(frac > 0)
        {
            markersZ[idx] = 1;
        }
        else
        {
            arrayZ[idx] = 0;
            markersZ[idx] = 0;
        }
    }
}

__global__ void CUDA_ResolveFrictionWithCollider(double* arrayX, double* arrayY, double* arrayZ, 
                                 double* colliderSdf, CUDA_Int3 size, CUDA_Vector3 gridSpacing, CUDA_Vector3 origin, 
                                 CUDA_Vector3 dataXOrigin, CUDA_Vector3 dataYOrigin, CUDA_Vector3 dataZOrigin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        double tmpX;
        double tmpY;
        double tmpZ;

        CUDA_Int3 idx3 = {i, j, k};
        CUDA_Vector3 pos = CUDA_GridIdxToPosition(idx3, gridSpacing, dataXOrigin);
        if(CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos) < 0)
        {
            CUDA_Vector3 coliiderVel = {0, 0, 0};
            CUDA_Vector3 vel = CUDA_SampleFaceCenteredGrid3(arrayX, arrayY, arrayZ, origin, gridSpacing, size, pos);
            CUDA_Vector3 gradient = CUDA_GradientArray3(colliderSdf, origin, gridSpacing, size, pos);
            double gradientVectorLen = CUDA_Vector3GetLength(gradient);
            if(gradientVectorLen * gradientVectorLen > 0.0)
            {
                CUDA_Vector3 normal = CUDA_Vector3GetNormalised(gradient);
                CUDA_Vector3 velr = {vel.x - coliiderVel.x, vel.y - coliiderVel.y, vel.z - coliiderVel.z};
                CUDA_Vector3 velt = CUDA_ApplyFriction(velr, normal, 1.0);
                CUDA_Vector3 velp = {velt.x + coliiderVel.x, velt.y + coliiderVel.y, velt.z + coliiderVel.z};
                tmpX = velp.x;
            }
            else
            {
                tmpX = coliiderVel.x;
            }

        }
        else
        {
            tmpX = arrayX[idx];
        }

        pos = CUDA_GridIdxToPosition(idx3, gridSpacing, dataYOrigin);
        if(CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos) < 0)
        {
            CUDA_Vector3 coliiderVel = {0, 0, 0};
            CUDA_Vector3 vel = CUDA_SampleFaceCenteredGrid3(arrayX, arrayY, arrayZ, origin, gridSpacing, size, pos);
            CUDA_Vector3 gradient = CUDA_GradientArray3(colliderSdf, origin, gridSpacing, size, pos);
            double gradientVectorLen = CUDA_Vector3GetLength(gradient);
            if(gradientVectorLen * gradientVectorLen > 0.0)
            {
                CUDA_Vector3 normal = CUDA_Vector3GetNormalised(gradient);
                CUDA_Vector3 velr = {vel.x - coliiderVel.x, vel.y - coliiderVel.y, vel.z - coliiderVel.z};
                CUDA_Vector3 velt = CUDA_ApplyFriction(velr, normal, 1.0);
                CUDA_Vector3 velp = {velt.x + coliiderVel.x, velt.y + coliiderVel.y, velt.z + coliiderVel.z};
                tmpY = velp.y;
            }
            else
            {
                tmpY = coliiderVel.y;
            }
        }
        else
        {
            tmpY = arrayY[idx];
        }

        pos = CUDA_GridIdxToPosition(idx3, gridSpacing, dataZOrigin);
        if(CUDA_SampleArray3(colliderSdf, origin, gridSpacing, size, pos) < 0)
        {
            CUDA_Vector3 coliiderVel = {0, 0, 0};
            CUDA_Vector3 vel = CUDA_SampleFaceCenteredGrid3(arrayX, arrayY, arrayZ, origin, gridSpacing, size, pos);
            CUDA_Vector3 gradient = CUDA_GradientArray3(colliderSdf, origin, gridSpacing, size, pos);
            double gradientVectorLen = CUDA_Vector3GetLength(gradient);
            if(gradientVectorLen * gradientVectorLen > 0.0)
            {
                CUDA_Vector3 normal = CUDA_Vector3GetNormalised(gradient);
                CUDA_Vector3 velr = {vel.x - coliiderVel.x, vel.y - coliiderVel.y, vel.z - coliiderVel.z};
                CUDA_Vector3 velt = CUDA_ApplyFriction(velr, normal, 1.0);
                CUDA_Vector3 velp = {velt.x + coliiderVel.x, velt.y + coliiderVel.y, velt.z + coliiderVel.z};
                tmpZ = velp.z;
            }
            else
            {
                tmpZ = coliiderVel.z;
            }
        }
        else
        {
            tmpZ = arrayZ[idx];
        }

        __syncthreads();

        arrayX[idx] = tmpX;
        arrayY[idx] = tmpY;
        arrayZ[idx] = tmpZ;
    }
}

__global__ void CUDA_ResolveCollisionsWithDomainBoundry(double* arrayX, double* arrayY, double* arrayZ, CUDA_Int3 size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        if(i == 0 || i == size.x - 1)
        {
            arrayX[idx] = 0;
        }
        if(j == 0 || j == size.y - 1)
        {
            arrayY[idx] = 0;
        }
        if(k == 0 || k == size.z - 1)
        {
            arrayZ[idx] = 0;
        }
    }
}

__global__ void CUDA_BuildColliderMarkers(int* markers, double* colliderSdf, CUDA_Int3 size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        if(colliderSdf[idx] < 0)
        {
            markers[idx] = 1;
        }
        else
        {
            markers[idx] = 0;
        }
    }
}

__global__ void CUDA_ResolveCollisionsWithCollider(double* arrayX, double* arrayY, double* arrayZ, double* colliderSdf, CUDA_Int3 size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = i + size.x * (j + size.y * k);

    if(i >= 0 && j >= 0 && k >= 0 && i < size.x && j < size.y && k < size.z)
    { 
        if (colliderSdf[idx] < 0) 
        {
            if (i > 0 && colliderSdf[(i - 1) + size.x * (j + size.y * k)] >= 0) 
            {
                arrayX[idx] = 0;
            }
            if (j > 0 && colliderSdf[i + size.x * ((j - 1) + size.y * k)] >= 0) 
            {
                arrayY[idx] = 0;
            }
            if (k > 0 && colliderSdf[i + size.x * (j + size.y * (k - 1))] >= 0) 
            {
                arrayZ[idx] = 0;
            }

            __syncthreads();

            if (i < size.x - 1 && colliderSdf[(i + 1) + size.x * (j + size.y * k)] >= 0) 
            {
                arrayX[(i + 1) + size.x * (j + size.y * k)] = 0;
            }

            if (j < size.y - 1 && colliderSdf[i + size.x * ((j + 1) + size.y * k)] >= 0) 
            {
                arrayY[i + size.x * ((j + 1) + size.y * k)] = 0;
            }

            if (k < size.z - 1 && colliderSdf[i + size.x * (j + size.y * (k + 1))] >= 0) 
            {
                arrayZ[i + size.x * (j + size.y * (k + 1))] = 0;
            }
        }
    }
}

CudaBlockedBoundryConditionSolver::CudaBlockedBoundryConditionSolver()
{

}

CudaBlockedBoundryConditionSolver::~CudaBlockedBoundryConditionSolver()
{

}

void CudaBlockedBoundryConditionSolver::ConstrainVelocity(FaceCenteredGrid3D& velocity, size_t depth)
{
    const auto& size = velocity.GetSize();
    const size_t vectorSize = size.x * size.y * size.z;

    auto colliderSdfPtr = GetColliderSdf().Serialize();
    const auto& colliderSdf = GetColliderSdf();

    auto& xData = velocity.GetDataXRef();
    auto& yData = velocity.GetDataYRef();
    auto& zData = velocity.GetDataZRef();

    dim3 dimGrid = CalculateDimGrid(size);
    dim3 dimBlock = dim3(THREADS_IN_X, THREADS_IN_Y, THREADS_IN_Z);

    CUDA_Int3 d_size = Vector3SizeToCUDA_Int3(size);
    CUDA_Vector3 d_gridSpacing = Vector3ToCUDA_Vector3(velocity.GetGridSpacing());
    CUDA_Vector3 d_origin = Vector3ToCUDA_Vector3(velocity.GetOrigin());
    CUDA_Vector3 d_dataXOrigin = Vector3ToCUDA_Vector3(velocity.GetDataXOrigin());
    CUDA_Vector3 d_dataYOrigin = Vector3ToCUDA_Vector3(velocity.GetDataYOrigin());
    CUDA_Vector3 d_dataZOrigin = Vector3ToCUDA_Vector3(velocity.GetDataZOrigin());

    double *h_xData, *h_yData, *h_zData, *h_colliderSdf;
    double *d_xData, *d_yData, *d_zData, *d_colliderSdf, *d_tmpX, *d_tmpY, *d_tmpZ;
    int *d_markersX, *d_markersY, *d_markersZ;

    h_xData = (double*)malloc(vectorSize * sizeof(double));
    h_yData = (double*)malloc(vectorSize * sizeof(double));
    h_zData = (double*)malloc(vectorSize * sizeof(double));
    h_colliderSdf = (double*)malloc(vectorSize * sizeof(double));

    cudaMalloc((void **)&d_xData, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_yData, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_zData, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_colliderSdf, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_tmpX, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_tmpY, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_tmpZ, vectorSize * sizeof(double));
    cudaMalloc((void **)&d_markersX, vectorSize * sizeof(int));
    cudaMalloc((void **)&d_markersY, vectorSize * sizeof(int));
    cudaMalloc((void **)&d_markersZ, vectorSize * sizeof(int));

     xData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        h_xData[i + size.x * (j + size.y * k)] = xData(i, j, k);
        h_yData[i + size.x * (j + size.y * k)] = yData(i, j, k);
        h_zData[i + size.x * (j + size.y * k)] = zData(i, j, k);
        h_colliderSdf[i + size.x * (j + size.y * k)] = colliderSdf(i, j, k);
    }); 

    cudaMemcpy(d_xData, h_xData, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yData, h_yData, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zData, h_zData, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colliderSdf, h_colliderSdf, vectorSize * sizeof(double), cudaMemcpyHostToDevice);
   

    CUDA_BuildMarkers<<<dimGrid, dimBlock>>>(d_xData, d_markersX, d_yData, d_markersY, d_zData, d_markersZ, 
                                 d_colliderSdf, d_size, d_gridSpacing, d_origin, 
                                 d_dataXOrigin, d_dataYOrigin, d_dataZOrigin);


    cudaMemcpy(d_tmpX, d_xData, vectorSize * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_tmpY, d_yData, vectorSize * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_tmpZ, d_zData, vectorSize * sizeof(double), cudaMemcpyDeviceToDevice);

    CUDA_ExtrapolateToRegion<<<dimGrid, dimBlock>>>(d_tmpX, d_markersX, depth, d_xData, size.x, size.y, size.z);
    CUDA_ExtrapolateToRegion<<<dimGrid, dimBlock>>>(d_tmpY, d_markersY, depth, d_yData, size.x, size.y, size.z);
    CUDA_ExtrapolateToRegion<<<dimGrid, dimBlock>>>(d_tmpZ, d_markersZ, depth, d_zData, size.x, size.y, size.z);

    CUDA_ResolveFrictionWithCollider<<<dimGrid, dimBlock>>>(d_xData, d_yData, d_zData, d_colliderSdf, 
                                                            d_size, d_gridSpacing, d_origin, 
                                                            d_dataXOrigin, d_dataYOrigin, d_dataZOrigin);

    CUDA_ResolveCollisionsWithDomainBoundry<<<dimGrid, dimBlock>>>(d_xData, d_yData, d_zData, d_size);
    CUDA_ResolveCollisionsWithCollider<<<dimGrid, dimBlock>>>(d_xData, d_yData, d_zData, d_colliderSdf, d_size);

    cudaMemcpy(h_xData, d_xData, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_yData, d_yData, vectorSize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zData, d_zData, vectorSize * sizeof(double), cudaMemcpyDeviceToHost); 

     xData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        xData(i, j, k) = h_xData[i + size.x * (j + size.y * k)];
        yData(i, j, k) = h_yData[i + size.x * (j + size.y * k)];
        zData(i, j, k) = h_zData[i + size.x * (j + size.y * k)];
    });  

    cudaFree(d_xData);
    cudaFree(d_yData);
    cudaFree(d_zData);
    cudaFree(d_colliderSdf);
    cudaFree(d_tmpX);
    cudaFree(d_tmpY);
    cudaFree(d_tmpZ);
    cudaFree(d_markersX);
    cudaFree(d_markersY);
    cudaFree(d_markersZ);
    free(h_xData);
    free(h_yData);
    free(h_zData);
    free(h_colliderSdf);
}

dim3 CudaBlockedBoundryConditionSolver::CalculateDimGrid(const Vector3<size_t> size)
{
    int blocksInX = (int)std::ceil(((double)size.x) / THREADS_IN_X);
    int blocksInY = (int)std::ceil(((double)size.y) / THREADS_IN_Y);
    int blocksInZ = (int)std::ceil(((double)size.z) / THREADS_IN_Z);
    return dim3(blocksInX, blocksInY, blocksInZ);
}

