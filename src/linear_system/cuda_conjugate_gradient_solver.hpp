#ifndef _CUDA_CONJUGATE_GRADIENT_SOLVER_HPP
#define _CUDA_CONJUGATE_GRADIENT_SOLVER_HPP

#include "linear_system_solver.hpp"
#include "cuda_blas.hpp"


class CudaConjugateGradientSolver : public LinearSystemSolver
{
    public:
        CudaConjugateGradientSolver(size_t maxNumberOfIterations, double tolerance);
        ~CudaConjugateGradientSolver();

        void Solve(LinearSystem* system) override;

    protected: 
        virtual void Preconditioner(double* a, double* b, CUDA_Int3 size, dim3 dimGrid, dim3 dimBlock);

    private:
        size_t _maxNumberOfIterations;
        double _tolerance;

        double* _d_x;
        double* _d_b;
        double* _d_ACenter;
        double* _d_ARight;
        double* _d_AUp;
        double* _d_AFront;
        double* _d_r;
        double* _d_d;
        double* _d_q;
        double* _d_s;
        double* _d_tmp;

        bool _wasMemoryAllocatedOnDevice;

        void FromHostToDevice(LinearSystem* system);
        void FromDeviceToHost(LinearSystem* system);
        void InitializeSolver(const Vector3<size_t>& size);
        void AllocateMemoryOnDevice(size_t vectorSize);
        void FreeDeviceMemory();
};



#endif // _CUDA_CONJUGATE_GRADIENT_SOLVER_HPP