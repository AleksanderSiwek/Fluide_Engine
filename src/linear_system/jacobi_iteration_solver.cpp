#include "jacobi_iteration_solver.hpp"

#include <iostream>

JacobiIterationSolver::JacobiIterationSolver(size_t maxNumberOfIterations, size_t toleranceCheckInterval, double tolerance)
    : _maxNumberOfIterations(maxNumberOfIterations), _toleranceCheckInterval(toleranceCheckInterval), _tolerance(tolerance)
{

}

JacobiIterationSolver::~JacobiIterationSolver()
{

}

void JacobiIterationSolver::Solve(LinearSystem* system)
{
    _xTemp.Resize(system->x.GetSize());

    for(size_t i = 0; i < _maxNumberOfIterations; i++)
    {
        Relax(system);
        if(i != 0 && i % _toleranceCheckInterval == 0)
        {
            if(CalculateTolerance(system->x) < _tolerance)
                break;
        }
    }
}

void JacobiIterationSolver::Relax(LinearSystem* system)
{
    Vector3<size_t> size = system->x.GetSize();

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                double r =
                    ((i > 0) ? system->A(i - 1, j, k).right * system->x(i - 1, j, k) : 0.0) +
                    ((i + 1 < size.x) ? system->A(i, j, k).right * system->x(i + 1, j, k) : 0.0) +
                    ((j > 0) ? system->A(i, j - 1, k).up * system->x(i, j - 1, k) : 0.0) +
                    ((j + 1 < size.y) ? system->A(i, j, k).up * system->x(i, j + 1, k) : 0.0) +
                    ((k > 0) ? system->A(i, j, k - 1).front * system->x(i, j, k - 1) : 0.0) +
                    ((k + 1 < size.z) ? system->A(i, j, k + 1).front * system->x(i, j, k + 1) : 0.0);

                _xTemp(i, j, k) = (system->b(i, j, k) - r) / system->A(i, j, k).center;
            }
        }
    }
    system->x.Swap(_xTemp);
}

double JacobiIterationSolver::CalculateTolerance(const SystemVector& x)
{
    Vector3<size_t> size = x.GetSize();

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                _xTemp(i, j, k) -= x(i, j, k);
            }
        }
    }

    return BLAS::L2Norm(_xTemp);
}
