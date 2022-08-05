#ifndef _CUDA_JACOBI_ITERATION_SOLVER_HPP
#define _CUDA_JACOBI_ITERATION_SOLVER_HPP

#include "linear_system_solver.hpp"
#include "cuda_blas.hpp"


class CudaJacobiIterationSolver : public LinearSystemSolver
{
    public:
        CudaJacobiIterationSolver(size_t maxNumberOfIterations, size_t residualCheckInterval, double tolerance);
        ~CudaJacobiIterationSolver();

        void Solve(LinearSystem* system) override;

    private:
        size_t _maxNumberOfIterations;
        size_t _toleranceCheckInterval;
        double _tolerance;
        size_t _iteration;

        bool _wasMemoryAllocatedOnDevice;

        double* _d_x;
        double* _d_b;
        double* _d_ACenter;
        double* _d_ARight;
        double* _d_AUp;
        double* _d_AFront;
        double* _d_residual;
        double* _d_xTemp;
        double* _d_tmp;

        void Relax(const Vector3<size_t> size);
        double CalculateTolerance(const Vector3<size_t> size);
        void Initialize(LinearSystem* system);
        void FromDeviceToHost(LinearSystem* system);
        void AllocateMemoryOnDevice(size_t vectorSize);
        void FreeDeviceMemory();
};


#endif // _CUDA_JACOBI_ITERATION_SOLVER_HPP