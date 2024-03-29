#include "blocked_boundry_condition_solver.hpp"
#include "../common/math_utils.hpp"
#include "../grid_systems/fluid_markers.hpp"

#define MAX_DISTANCE 1.7976931348623157E+30


BlockedBoundryConditionSolver::BlockedBoundryConditionSolver()
{

}

BlockedBoundryConditionSolver::~BlockedBoundryConditionSolver()
{

}

void BlockedBoundryConditionSolver::ConstrainVelocity(FaceCenteredGrid3D& velocity, size_t depth)
{
    const auto& size = velocity.GetSize();
    const auto& gridSpacing = velocity.GetGridSpacing();
    const auto& origin = velocity.GetOrigin();
    //UpdateCollider(size, gridSpacing, origin);

    auto& xData = velocity.GetDataXRef();
    auto& yData = velocity.GetDataYRef();
    auto& zData = velocity.GetDataZRef();

    Array3<double> xTemp(xData.GetSize());
    Array3<double> yTemp(yData.GetSize());
    Array3<double> zTemp(zData.GetSize());
    Array3<int> xMarker(xData.GetSize(), 1);
    Array3<int> yMarker(yData.GetSize(), 1);
    Array3<int> zMarker(zData.GetSize(), 1);

    const auto& colliderSdf = GetColliderSdf();

    xMarker.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> pos = velocity.GetXPos(i, j, k);
        double phi0 = colliderSdf.Sample(pos - Vector3<double>(0.5 * gridSpacing.x, 0.0, 0.0));
        double phi1 = colliderSdf.Sample(pos + Vector3<double>(0.5 * gridSpacing.x, 0.0, 0.0));
        double frac = FractionInsideSdf(phi0, phi1);
        frac = 1 - Clamp(frac, 1.0, 0.0);
        if(frac > 0.0)
        {
            xMarker(i, j, k) = 1;
        }
        else
        {
            Vector3<double> colliderVelocity = 0;
            xData(i, j, k) = colliderVelocity.x; // TO DO: This should be collider velocity btw
            xMarker(i, j, k) = 0;
        }
    });

    yMarker.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> pos = velocity.GetYPos(i, j, k);
        double phi0 = colliderSdf.Sample(pos - Vector3<double>(0.0, 0.5 * gridSpacing.y, 0.0));
        double phi1 = colliderSdf.Sample(pos + Vector3<double>(0.0, 0.5 * gridSpacing.y, 0.0));
        double frac = FractionInsideSdf(phi0, phi1);
        frac = 1 - Clamp(frac, 1.0, 0.0);

        if(frac > 0.0)
        {
            yMarker(i, j, k) = 1;
        }
        else
        {
            Vector3<double> colliderVelocity = 0;
            yData(i, j, k) = colliderVelocity.y; // TO DO: This should be collider velocity btw
            yMarker(i, j, k) = 0;
        }
    });

    zMarker.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> pos = velocity.GetZPos(i, j, k);
        double phi0 = colliderSdf.Sample(pos - Vector3<double>(0.0, 0.0, 0.5 * gridSpacing.z));
        double phi1 = colliderSdf.Sample(pos + Vector3<double>(0.0, 0.0, 0.5 * gridSpacing.z));
        double frac = FractionInsideSdf(phi0, phi1);
        frac = 1 - Clamp(frac, 1.0, 0.0);

        if(frac > 0.0)
        {
            zMarker(i, j, k) = 1;
        }
        else
        {
            Vector3<double> colliderVelocity = 0;
            zData(i, j, k) = colliderVelocity.z; // TO DO: This should be collider velocity btw
            zMarker(i, j, k) = 0;
        }
    });

    const auto prevX(velocity.GetDataX());
    const auto prevY(velocity.GetDataY());
    const auto prevZ(velocity.GetDataZ());
    ExtrapolateToRegion(prevX, xMarker, depth, xData);
    ExtrapolateToRegion(prevY, yMarker, depth, yData);
    ExtrapolateToRegion(prevZ, zMarker, depth, zData);

    xTemp.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> position = velocity.GetXPos(i, j, k);
        if(colliderSdf.Sample(position) < 0)
        {
            Vector3<double> coliderVel = 0; // TO DO
            Vector3<double> vel = velocity.Sample(position);
            Vector3<double> g = colliderSdf.Gradient(position);
            if(g.GetLength() * g.GetLength() > 0.0)
            {
                Vector3<double> n = g.GetNormalized();
                Vector3<double> velr = vel - coliderVel;
                Vector3<double> velt = ApplyFriction(velr, n, 0.0);
                Vector3<double> velp = velt + coliderVel;
                xTemp(i, j, k) = velp.x;
            }
            else
            {
                xTemp(i, j, k) = coliderVel.x;
            }
        }
        else
        {
            xTemp(i, j, k) = xData(i, j, k);   
        }
    });

    yTemp.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> position = velocity.GetYPos(i, j, k);
        if(colliderSdf.Sample(position) < 0)
        {
            Vector3<double> coliderVel = 0; // TO DO
            Vector3<double> vel = velocity.Sample(position);
            Vector3<double> g = colliderSdf.Gradient(position);
            if(g.GetLength() * g.GetLength() > 0.0)
            {
                Vector3<double> n = g.GetNormalized();
                Vector3<double> velr = vel - coliderVel;
                Vector3<double> velt = ApplyFriction(velr, n, 0.0);
                Vector3<double> velp = velt + coliderVel;
                yTemp(i, j, k) = velp.y;
            }
            else
            {
                yTemp(i, j, k) = coliderVel.y;
            }
        }
        else
        {
            yTemp(i, j, k) = yData(i, j, k);   
        }
    });

    zTemp.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        Vector3<double> position = velocity.GetZPos(i, j, k);
        if(colliderSdf.Sample(position) < 0)
        {
            Vector3<double> coliderVel = 0; // TO DO
            Vector3<double> vel = velocity.Sample(position);
            Vector3<double> g = colliderSdf.Gradient(position);
            if(g.GetLength() * g.GetLength() > 0.0)
            {
                Vector3<double> n = g.GetNormalized();
                Vector3<double> velr = vel - coliderVel;
                Vector3<double> velt = ApplyFriction(velr, n, 0.0);
                Vector3<double> velp = velt + coliderVel;
                zTemp(i, j, k) = velp.z;
            }
            else
            {
                zTemp(i, j, k) = coliderVel.z;
            }
        }
        else
        {
            zTemp(i, j, k) = zData(i, j, k);   
        }
    });

    xData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        xData(i, j, k) = xTemp(i, j, k);
    });
    yData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        yData(i, j, k) = yTemp(i, j, k);
    });
    zData.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        zData(i, j, k) = zTemp(i, j, k);
    });

    for (size_t k = 0; k < xData.GetSize().z; ++k) 
    {
        for (size_t j = 0; j < xData.GetSize().y; ++j) 
        {
            xData(0, j, k) = 0;
        }
    }

    for (size_t k = 0; k < xData.GetSize().z; ++k) 
    {
        for (size_t j = 0; j < xData.GetSize().y; ++j) 
        {
            xData(xData.GetSize().x - 1, j, k) = 0;
        }
    }

    for (size_t k = 0; k < yData.GetSize().z; ++k) 
    {
        for (size_t i = 0; i < yData.GetSize().x; ++i) 
        {
            yData(i, 0, k) = 0;
        }
    }

    for (size_t k = 0; k < yData.GetSize().z; ++k) 
    {
        for (size_t i = 0; i < yData.GetSize().x; ++i) 
        {
            yData(i, yData.GetSize().y - 1, k) = 0;
        }
    }

    for (size_t j = 0; j < zData.GetSize().y; ++j) 
    {
        for (size_t i = 0; i < zData.GetSize().x; ++i) 
        {
            zData(i, j, 0) = 0;
        }
    }

    for (size_t j = 0; j < zData.GetSize().y; ++j) 
    {
        for (size_t i = 0; i < zData.GetSize().x; ++i) 
        {
            zData(i, j, zData.GetSize().z - 1) = 0;
        }
    }

    Array3<enum FluidMarker> markers;
    markers.Resize(size);
    markers.ParallelForEachIndex([&](size_t i, size_t j, size_t k) 
    {
        if (colliderSdf(i, j, k) < 0) 
        {
            markers(i, j, k) = BOUNDRY_MARK;
        } 
        else 
        {
            markers(i, j, k) = FLUID_MARK;
        }
    });

    // TO DO change 0 to collider Velocity on  respective axis
    markers.ForEachIndex([&](size_t i, size_t j, size_t k)
    {
        if (markers(i, j, k) == BOUNDRY_MARK) 
        {
            if (i > 0 && markers(i - 1, j, k) == FLUID_MARK) 
            {
                xData(i, j, k) = 0;
            }
            if (i < size.x - 1 && markers(i + 1, j, k) == FLUID_MARK) 
            {
                xData(i + 1, j, k) = 0;
            }
            if (j > 0 && markers(i, j - 1, k) == FLUID_MARK) 
            {
                yData(i, j, k) = 0;
            }
            if (j < size.y - 1 && markers(i, j + 1, k) == FLUID_MARK) 
            {
                yData(i, j + 1, k) = 0;
            }
            if (k > 0 && markers(i, j, k - 1) == FLUID_MARK) 
            {
                zData(i, j, k) = 0;
            }
            if (k < size.z - 1 && markers(i, j, k + 1) == FLUID_MARK) 
            {
                zData(i, j, k + 1) = 0;
            }
        }
    });
}

void BlockedBoundryConditionSolver::UpdateCollider(Vector3<size_t> size, Vector3<double> gridSpacing, Vector3<double> gridOrigin)
{
    // colliderSdf.Resize(size);
    // colliderSdf.ParallelFill(MAX_DISTANCE);
    // colliderSdf.SetGridSpacing(gridSpacing);
    // colliderSdf.SetOrigin(gridOrigin);

    // _colliderVel.Resize(size);
    // _colliderVel.ParallelFill(Vector3<double>(0, 0, 0));
}

double BlockedBoundryConditionSolver::FractionInsideSdf(double phi0, double phi1) const
{
    if (phi0 < 0.0 && phi1 < 0.0) 
    {
        return 1;
    } 
    else if (phi0 < 0.0 && !(phi1 < 0.0)) 
    {
        return phi0 / (phi0 - phi1);
    } 
    else if (!(phi0 < 0.0) && phi1 < 0.0) 
    {
        return phi1 / (phi1 - phi0);
    } 
    else 
    {
        return 0;
    }
}

Vector3<double> BlockedBoundryConditionSolver::ApplyFriction(Vector3<double> vel, Vector3<double> normal, double frictionCoeddicient)
{
    Vector3<double> velt = vel.Project(normal);
    if(velt.GetLength() * velt.GetLength() > 0.0)
    {
        double veln = std::max(-vel.Dot(normal), 0.0);
        velt != std::max(1.0 - frictionCoeddicient * veln / velt.GetLength(), 0.0);
    }
    return velt;
}