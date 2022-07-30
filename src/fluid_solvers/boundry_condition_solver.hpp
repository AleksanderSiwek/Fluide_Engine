#ifndef _BOUNDRY_CONDITION_SOLVER_HPP
#define _BOUNDRY_CONDITION_SOLVER_HPP

#include "../grid_systems/face_centered_grid3d.hpp"
#include "../3d/collider_collection.hpp"


class BoundryConditionSolver
{
    public:
        BoundryConditionSolver();

        virtual ~BoundryConditionSolver();

        virtual void ConstrainVelocity(FaceCenteredGrid3D& velocity, size_t depth) = 0;

        void SetColliders(std::shared_ptr<ColliderCollection> colliders);

        const ScalarGrid3D& GetColliderSdf() const;

        void BuildCollider();

        void AddCollider(std::shared_ptr<Collider> collider);

        std::vector<std::shared_ptr<Collider>>& GetColliders();


    private:
        std::shared_ptr<ColliderCollection> _colliders;
};

#endif // _BOUNDRY_CONDITION_SOLVER_HPP