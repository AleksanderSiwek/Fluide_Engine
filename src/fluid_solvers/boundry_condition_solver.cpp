#include "boundry_condition_solver.hpp"


BoundryConditionSolver::BoundryConditionSolver()
{
    
}

BoundryConditionSolver::~BoundryConditionSolver()
{

}

void BoundryConditionSolver::SetColliders(std::shared_ptr<ColliderCollection> colliders)
{
    _colliders = colliders;
}

const ScalarGrid3D& BoundryConditionSolver::GetColliderSdf() const
{
    return _colliders->GetSdf();
}

void BoundryConditionSolver::BuildCollider()
{
    _colliders->BuildSdf();
}

std::vector<std::shared_ptr<Collider>>& BoundryConditionSolver::GetColliders()
{
    return _colliders->GetColliders();
}

void BoundryConditionSolver::AddCollider(std::shared_ptr<Collider> collider)
{
    _colliders->AddCollider(collider);
}
