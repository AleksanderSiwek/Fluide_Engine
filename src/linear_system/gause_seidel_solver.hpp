#ifndef _GAUS_SEIDEL_SOLVER_HPP
#define _GAUS_SEIDEL_SOLVER_HPP

#include "linear_system_solver.hpp"


class GausSeidelSolver : public LinearSystemSolver
{
    public:
        GausSeidelSolver();
        
        ~GausSeidelSolver();

        void Solve(LinearSystem* system) override;
        
    private:
};


#endif