#include "jacobi_iteration_solver.hpp"


JacobiIterationSolver::JacobiIterationSolver(size_t maxNumberOfIterations, size_t toleranceCheckInterval, double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations), _toleranceCheckInterval(toleranceCheckInterval), _tolerance(tolerance)
{

}

JacobiIterationSolver::~JacobiIterationSolver()
{

}

void JacobiIterationSolver::Solve(LinearSystem* system)
{
    _xTemp.Clear();
    _residual.Clear();

    _xTemp.Resize(system->x.GetSize());
    _residual.Resize(system->x.GetSize());

    for(size_t i = 0; i < _maxNumberOfIterations; i++)
    {
        Relax(system);
        _xTemp.Swap(system->x);
        
        if(i != 0 && i % _toleranceCheckInterval == 0)
        {
            if(CalculateTolerance(system) < _tolerance)
            {
                break;
            }
        }
    }
}

void JacobiIterationSolver::Relax(LinearSystem* system)
{
    const auto& size = system->x.GetSize();
    const auto& A = system->A;
    const auto& b = system->b;
    auto& x = system->x;

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                double r =
                    ((i > 0) ? A(i - 1, j, k).right * x(i - 1, j, k) : 0.0) +
                    ((i + 1 < size.x) ? A(i, j, k).right * x(i + 1, j, k) : 0.0) +
                    ((j > 0) ? A(i, j - 1, k).up * x(i, j - 1, k) : 0.0) +
                    ((j + 1 < size.y) ? A(i, j, k).up * x(i, j + 1, k) : 0.0) +
                    ((k > 0) ? A(i, j, k - 1).front * x(i, j, k - 1) : 0.0) +
                    ((k + 1 < size.z) ? A(i, j, k).front * x(i, j, k + 1) : 0.0);

                _xTemp(i, j, k) = (b(i, j, k) - r) / A(i, j, k).center;
            }
        }
    }
}

double JacobiIterationSolver::CalculateTolerance(LinearSystem* system)
{
    BLAS::Residual(system->A, system->x, system->b, &_residual);
    return BLAS::L2Norm(_residual);
}
