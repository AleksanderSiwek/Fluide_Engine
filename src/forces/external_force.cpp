#include "external_force.hpp"


void ExternalForce::ApplyExternalForce(FaceCenteredGrid3D& velGrid, const double timeIntervalInSeconds)
{
    auto& x = velGrid.GetDataXRef();
    auto& y = velGrid.GetDataYRef();
    auto& z = velGrid.GetDataZRef();

    x.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        const auto& position = velGrid.GetXPos(i, j, k);
        const auto& forceVector = Sample(position);
        x(i, j, k) += timeIntervalInSeconds * forceVector.x;
    });

    y.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        const auto& position = velGrid.GetYPos(i, j, k);
        const auto& forceVector = Sample(position);
        y(i, j, k) += timeIntervalInSeconds * forceVector.y;
    });

    z.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        const auto& position = velGrid.GetZPos(i, j, k);
        const auto& forceVector = Sample(position);
        z(i, j, k) += timeIntervalInSeconds * forceVector.z;
    });
}