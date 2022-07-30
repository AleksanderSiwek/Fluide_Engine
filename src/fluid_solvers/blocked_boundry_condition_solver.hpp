#ifndef _BLOCKED_BOUNDRY_CONDITION_SOLVER_HPP
#define _BLOCKED_BOUNDRY_CONDITION_SOLVER_HPP

#include "boundry_condition_solver.hpp"
#include "../3d/scalar_grid3d.hpp"


class BlockedBoundryConditionSolver : public BoundryConditionSolver
{
    public:
        BlockedBoundryConditionSolver();
        ~BlockedBoundryConditionSolver();
        
        void ConstrainVelocity(FaceCenteredGrid3D& velocity, size_t depth) override;

    private:
        Array3<Vector3<double>> _colliderVel;

        void UpdateCollider(Vector3<size_t> size, Vector3<double> gridSpacing, Vector3<double> gridOrigin);
        double FractionInsideSdf(double phi0, double phi1) const;
        Vector3<double> ApplyFriction(Vector3<double> vel, Vector3<double> normal, double frictionCoeddicient);
};


#endif // _BLOCKED_BOUNDRY_CONDITION_SOLVER_HPP
