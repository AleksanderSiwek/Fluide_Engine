#ifndef _BOUNDRY_CONDITION_SOLVER_HPP
#define _BOUNDRY_CONDITION_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"


class BoundryConditionSolver
{
    public:
        BoundryConditionSolver();

        virtual ~BoundryConditionSolver();

        virtual void ConstrainVelocity(FaceCenteredGrid3D& velocity, size_t depth) = 0;
};

#endif // _BOUNDRY_CONDITION_SOLVER_HPP