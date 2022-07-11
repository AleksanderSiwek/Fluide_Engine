#include "external_force.hpp"


void ExternalForce::ApplyExternalForce(FaceCenteredGrid3D& velGrid, const double timeIntervalInSeconds)
{
    velGrid.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        const auto& position = velGrid.GetCellCenterPos(i, j, k);
        const auto& forceVector = Sample(position);
        velGrid.x(i, j, k) += timeIntervalInSeconds * forceVector.x;
        velGrid.y(i, j, k) += timeIntervalInSeconds * forceVector.y;
        velGrid.z(i, j, k) += timeIntervalInSeconds * forceVector.z;
    });
}