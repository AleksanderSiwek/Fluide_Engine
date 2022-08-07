#ifndef _CONJUGATE_GRADIENT_SOLVER_HPP
#define _CONJUGATE_GRADIENT_SOLVER_HPP

#include "linear_system_solver.hpp"
#include "blas.hpp"


class ConjugateGradientSolver : public LinearSystemSolver
{
    public:
        ConjugateGradientSolver(size_t maxNumberOfIterations, double tolerance);
        ~ConjugateGradientSolver();

        void Solve(LinearSystem* system) override;

    protected: 
        virtual void Preconditioner(SystemVector& a, SystemVector& b);

    private:
        size_t _maxNumberOfIterations;
        double _tolerance;

        SystemVector _r;
        SystemVector _d;
        SystemVector _q;
        SystemVector _s;
        SystemVector _dPre;
        SystemVector _yPre;

        void InitializeSolver(LinearSystem* system);
};



#endif // _CONJUGATE_GRADIENT_SOLVER_HPP