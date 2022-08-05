#include "conjugate_gradient_solver.hpp"

ConjugateGradientSolver::ConjugateGradientSolver(size_t maxNumberOfIterations, double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations), _tolerance(tolerance)
{

}

ConjugateGradientSolver::~ConjugateGradientSolver()
{

}

void ConjugateGradientSolver::Solve(LinearSystem* system)
{
    const auto& size = system->x.GetSize();

    InitializeSolver(size);

    BLAS::Residual(system->A, system->x, system->b, &_r);
    Preconditioner(_r, _d);

    double sigmaNew = BLAS::Dot(_r, _d);
    size_t iteration = 0 ;
    bool trigger = false;
    while(sigmaNew > _tolerance * _tolerance && iteration < _maxNumberOfIterations)
    {
        BLAS::MatrixVectorMultiplication(system->A, _d, &_q);
        double alpha = sigmaNew / BLAS::Dot(_d, _q);
        BLAS::AXpY(alpha, _d, system->x, &(system->x));

        if(trigger || (iteration % 50 == 0 && iteration > 0))
        {
            BLAS::Residual(system->A, system->x, system->b, &_r);
            trigger = false;
        }
        else
        {
            BLAS::AXpY(-alpha, _q, _r, &_r);
        }

        Preconditioner(_r, _s);

        double sigmaOld = sigmaNew;
        sigmaNew = BLAS::Dot(_r, _s);
        if(sigmaNew > sigmaOld)
        {
            trigger = true;
        }

        double beta = sigmaNew / sigmaOld;
        BLAS::AXpY(beta, _d, _s, &_d);
        iteration++;
    }
}

void ConjugateGradientSolver::Preconditioner(SystemVector& a, SystemVector& b)
{
    b.ParallelFill(a);
}

void ConjugateGradientSolver::InitializeSolver(const Vector3<size_t>& size)
{
    _r.Resize(size);
    _d.Resize(size);
    _q.Resize(size);
    _s.Resize(size);

    _r.ParallelFill(0);
    _d.ParallelFill(0);
    _q.ParallelFill(0);
    _s.ParallelFill(0);
}