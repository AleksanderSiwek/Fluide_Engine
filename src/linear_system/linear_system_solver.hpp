#ifndef _LINEAR_SYSTEM_SOLVER_HPP
#define _LINEAR_SYSTEM_SOLVER_HPP


#include "linear_system.hpp"

class LinearSystemSolver
{
    public:
        LinearSystemSolver();
        virtual ~LinearSystemSolver() = default;

        virtual void Solve(LinearSystem* system) = 0;

    private:
    
};

#endif // _LINEAR_SYSTEM_SOLVER_HPP