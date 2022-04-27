#ifndef _JACOBI_ITERATION_SOLVER_HPP
#define _JACOBI_ITERATION_SOLVER_HPP

#include "linear_system_solver.hpp"
#include "blas.hpp"


class JacobiIterationSolver : public LinearSystemSolver
{
    public:
        JacobiIterationSolver(size_t maxNumberOfIterations, size_t residualCheckInterval, double tolerance);
        ~JacobiIterationSolver();

        void Solve(LinearSystem* system) override;

    private:
        size_t _maxNumberOfIterations;
        size_t _toleranceCheckInterval;
        double _tolerance;

        SystemVector _xTemp;
        SystemVector _residual;

        void Relax(LinearSystem* system);
        double CalculateTolerance(LinearSystem* system);
};


#endif // _JACOBI_ITERATION_SOLVER_HPP