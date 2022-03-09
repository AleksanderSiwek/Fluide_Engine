#ifndef _BOUNDRY_CONDITION_SOLVER_HPP
#define _BOUNDRY_CONDITION_SOLVER_HPP


class BoundryConditionSolver
{
    public:
        BoundryConditionSolver();

        virtual ~BoundryConditionSolver();

        virtual void Solve() = 0;
};

#endif // _BOUNDRY_CONDITION_SOLVER_HPP