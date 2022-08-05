#ifndef _CUDA_BACKWARD_EULER_DIFFUSION_SOLVER_HPP
#define _CUDA_BACKWARD_EULER_DIFFUSION_SOLVER_HPP

#include <cuda_runtime.h>

#include "boundry_condition_solver.hpp"
#include "../3d/scalar_grid3d.hpp"


class CudaBlockedBoundryConditionSolver : public BoundryConditionSolver
{
    public: 
        CudaBlockedBoundryConditionSolver();

        ~CudaBlockedBoundryConditionSolver();

        void ConstrainVelocity(FaceCenteredGrid3D& velocity, size_t depth) override;

    private:
        const int THREADS_IN_X = 4;
        const int THREADS_IN_Y = 4;
        const int THREADS_IN_Z = 4;

        Array3<Vector3<double>> _colliderVel;

        dim3 CalculateDimGrid(const Vector3<size_t> size);
};


#endif // _CUDA_BACKWARD_EULER_DIFFUSION_SOLVER_HPP