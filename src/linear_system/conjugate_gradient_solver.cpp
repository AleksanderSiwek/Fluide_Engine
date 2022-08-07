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

    InitializeSolver(system);

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
    // NULL type preconditioner
    b.ParallelFill(a);

    // TO DO ? ICCG - Incomplete Cholesky conjugate gradient
}

void ConjugateGradientSolver::InitializeSolver(LinearSystem* system)
{
    const auto& size = system->A.GetSize();

    _r.Resize(size, 0);
    _d.Resize(size, 0);
    _q.Resize(size, 0);
    _s.Resize(size, 0);

    // TO DO ? ICCG - Incomplete Cholesky conjugate gradient
    // _dPre.Resize(size, 0);
    // _yPre.Resize(size, 0);

    // system->A.ForEachIndex([&](size_t i, size_t j, size_t k)
    // {
    //     double denom = 
    //         system->A(i, j, k).center -
    //         ((i > 0) ? (system->A(i - 1, j, k).right) * (system->A(i - 1, j, k).right) * _dPre(i - 1, j, k) : 0.0) -
    //         ((j > 0) ? (system->A(i, j - 1, k).up) * (system->A(i, j - 1, k).up) * _dPre(i, j - 1, k) : 0.0) -
    //         ((k > 0) ? (system->A(i, j, k - 1).front) * (system->A(i, j, k - 1).front) * _dPre(i, j, k - 1): 0.0);

    //     if (std::fabs(denom) > 0.0) 
    //     {
    //         _dPre(i, j, k) = 1.0 / denom;
    //     } 
    //     else 
    //     {
    //         _dPre(i, j, k) = 0.0;
    //     }
    // });
}