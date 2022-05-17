#include "blocked_boundry_condition_solver.hpp"
#include "../common/math_utils.hpp"

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
    // const auto& gridSpacing = velocity.GetGridSpacing();
    // const auto& origin = velocity.GetOrigin();
    //UpdateCollider(size, gridSpacing, origin);

    auto& xData = velocity.GetDataXRef();
    auto& yData = velocity.GetDataYRef();
    auto& zData = velocity.GetDataZRef();

    // Array3<double> xTemp(xData.GetSize());
    // Array3<double> yTemp(yData.GetSize());
    // Array3<double> zTemp(zData.GetSize());
    // Array3<int> xMarker(xData.GetSize(), 1);
    // Array3<int> yMarker(yData.GetSize(), 1);
    // Array3<int> zMarker(zData.GetSize(), 1);

    // for(size_t i = 0; i < size.x; i++)
    // {
    //     for(size_t j = 0; j < size.y; j++)
    //     {
    //         for(size_t k = 0; k < size.z; k++)
    //         {
    //             Vector3<double> pos = velocity.GetXPos(i, j, k);
    //             double phi0 = _colliderSdf.Sample(pos - Vector3<double>(0.5 * gridSpacing.x, 0.0, 0.0));
    //             double phi1 = _colliderSdf.Sample(pos + Vector3<double>(0.5 * gridSpacing.x, 0.0, 0.0));
    //             double frac = FractionInsideSdf(phi0, phi1);
    //             frac = 1.0 - Clamp(frac, 0.0, 1.0);

    //             if(frac > 0.0)
    //             {
    //                 xMarker(i, j, k) = 1;
    //             }
    //             else
    //             {
    //                 xData(i, j, k) = 0; // This should be collider velocity btw
    //                 xMarker(i, j, k) = 0;
    //             }
    //         }
    //     }
    // }

    // for(size_t i = 0; i < size.x; i++)
    // {
    //     for(size_t j = 0; j < size.y; j++)
    //     {
    //         for(size_t k = 0; k < size.z; k++)
    //         {
    //             Vector3<double> pos = velocity.GetYPos(i, j, k);
    //             double phi0 = _colliderSdf.Sample(pos - Vector3<double>(0.0, 0.5 * gridSpacing.y, 0.0));
    //             double phi1 = _colliderSdf.Sample(pos + Vector3<double>(0.0, 0.5 * gridSpacing.y, 0.0));
    //             double frac = FractionInsideSdf(phi0, phi1);
    //             frac = 1.0 - Clamp(frac, 0.0, 1.0);

    //             if(frac > 0.0)
    //             {
    //                 yMarker(i, j, k) = 1;
    //             }
    //             else
    //             {
    //                 yData(i, j, k) = 0; // This should be collider velocity btw
    //                 yMarker(i, j, k) = 0;
    //             }
    //         }
    //     }
    // }

    // for(size_t i = 0; i < size.x; i++)
    // {
    //     for(size_t j = 0; j < size.y; j++)
    //    {
    //         for(size_t k = 0; k < size.z; k++)
    //         {
    //             Vector3<double> pos = velocity.GetZPos(i, j, k);
    //             double phi0 = _colliderSdf.Sample(pos - Vector3<double>(0.0, 0.0, 0.5 * gridSpacing.z));
    //             double phi1 = _colliderSdf.Sample(pos + Vector3<double>(0.0, 0.0, 0.5 * gridSpacing.z));
    //             double frac = FractionInsideSdf(phi0, phi1);
    //             frac = 1.0 - Clamp(frac, 0.0, 1.0);

    //             if(frac > 0.0)
    //             {
    //                 zMarker(i, j, k) = 1;
    //             }
    //             else
    //             {
    //                 zData(i, j, k) = 0; // This should be collider velocity btw
    //                 zMarker(i, j, k) = 0;
    //             }
    //         }
    //     }
    // }

    // Array3<double> prevX(velocity.GetDataX());
    // Array3<double> prevY(velocity.GetDataY());
    // Array3<double> prevZ(velocity.GetDataZ());
    // ExtrapolateToRegion(prevX, xMarker, depth, xData);
    // ExtrapolateToRegion(prevY, yMarker, depth, yData);
    // ExtrapolateToRegion(prevZ, zMarker, depth, zData);

    // for(size_t i = 0; i < size.x; i++)
    // {
    //         for(size_t j = 0; j < size.y; j++)
    //     {
    //         for(size_t k = 0; k < size.z; k++)
    //         {
    //             Vector3<double> position = velocity.GetXPos(i, j, k);
    //             if(_colliderSdf(i, j, k) < 0)
    //             {
    //                 Vector3<double> coliderVel = 0; // TO DO
    //                 Vector3<double> vel = velocity.Sample(position);
    //                 Vector3<double> g = _colliderSdf.Gradient(position); // TO DO collider SDF gradient
    //                 if(g.GetLength() * g.GetLength() > 0.0)
    //                 {
    //                     Vector3<double> n = g.GetNormalized();
    //                     Vector3<double> velr = vel - coliderVel;
    //                     Vector3<double> velt = ApplyFriction(velr, n, 1.0);
    //                     Vector3<double> velp = velt + coliderVel;
    //                     xTemp(i, j, k) = velp.x;
    //                 }
    //                 else
    //                 {
    //                     xTemp(i, j, k) = xData(i, j, k);
    //                 }
    //             }
            
    //         }
    //     }
    // }

    // for(size_t i = 0; i < size.x; i++)
    // {
    //         for(size_t j = 0; j < size.y; j++)
    //     {
    //         for(size_t k = 0; k < size.z; k++)
    //         {
    //             Vector3<double> position = velocity.GetYPos(i, j, k);
    //             if(_colliderSdf(i, j, k) < 0)
    //             {
    //                 Vector3<double> coliderVel = 0; // TO DO
    //                 Vector3<double> vel = velocity.Sample(position);
    //                 Vector3<double> g = _colliderSdf.Gradient(position); // TO DO collider SDF gradient
    //                 if(g.GetLength() * g.GetLength() > 0.0)
    //                 {
    //                     Vector3<double> n = g.GetNormalized();
    //                     Vector3<double> velr = vel - coliderVel;
    //                     Vector3<double> velt = ApplyFriction(velr, n, 1.0);
    //                     Vector3<double> velp = velt + coliderVel;
    //                     yTemp(i, j, k) = velp.y;
    //                 }
    //                 else
    //                 {
    //                     yTemp(i, j, k) = yData(i, j, k);
    //                 }
    //             }
            
    //         }
    //     }
    // }

    // for(size_t i = 0; i < size.x; i++)
    // {
    //     for(size_t j = 0; j < size.y; j++)
    //     {
    //         for(size_t k = 0; k < size.z; k++)
    //         {
    //             Vector3<double> position = velocity.GetZPos(i, j, k);
    //             if(_colliderSdf(i, j, k) < 0)
    //             {
    //                 Vector3<double> coliderVel = 0; // TO DO
    //                 Vector3<double> vel = velocity.Sample(position);
    //                 Vector3<double> g = _colliderSdf.Gradient(position); // TO DO collider SDF gradient
    //                 if(g.GetLength() * g.GetLength() > 0.0)
    //                 {
    //                     Vector3<double> n = g.GetNormalized();
    //                     Vector3<double> velr = vel - coliderVel;
    //                     Vector3<double> velt = ApplyFriction(velr, n, 1.0);
    //                     Vector3<double> velp = velt + coliderVel;
    //                     zTemp(i, j, k) = velp.z;
    //                 }
    //                 else
    //                 {
    //                     zTemp(i, j, k) = zData(i, j, k);
    //                 }
    //             }
    //         }
    //     }
    // }

    // for(size_t i = 0; i < size.x; i++)
    // {
    //     for(size_t j = 0; j < size.y; j++)
    //     {
    //         for(size_t k = 0; k < size.z; k++)
    //         {
    //             xData(i, j, k) = xTemp(i, j, k);
    //             yData(i, j, k) = yTemp(i, j, k);
    //             zData(i, j, k) = zTemp(i, j, k);
    //         }
    //     }
    // }


    for (size_t k = 0; k < size.z; ++k) 
    {
        for (size_t j = 0; j < size.y; ++j) 
        {
            xData(0, j, k) = 0;
        }
    }

    for (size_t k = 0; k < size.z; ++k) 
    {
        for (size_t j = 0; j <size.y; ++j) 
        {
            xData(size.x - 1, j, k) = 0;
        }
    }

    for (size_t k = 0; k < size.z; ++k) 
    {
        for (size_t i = 0; i < size.x; ++i) 
        {
            yData(i, 0, k) = 0;
        }
    }

    for (size_t k = 0; k < size.z; ++k) 
    {
        for (size_t i = 0; i < size.x; ++i) 
        {
            yData(i, size.y - 1, k) = 0;
        }
    }

    for (size_t j = 0; j < size.y; ++j) 
    {
        for (size_t i = 0; i < size.x; ++i) 
        {
            zData(i, j, 0) = 0;
        }
    }

    for (size_t j = 0; j < size.y; ++j) 
    {
        for (size_t i = 0; i < size.x; ++i) 
        {
            zData(i, j, size.z - 1) = 0;
        }
    }
}

void BlockedBoundryConditionSolver::UpdateCollider(Vector3<size_t> size, Vector3<double> gridSpacing, Vector3<double> gridOrigin)
{
    _colliderSdf.Resize(size);
    _colliderSdf.Fill(MAX_DISTANCE);
    _colliderSdf.SetGridSpacing(gridSpacing);
    _colliderSdf.SetOrigin(gridOrigin);

    _colliderVel.Resize(size);
    _colliderVel.Fill(Vector3<double>(0, 0, 0));
}

double BlockedBoundryConditionSolver::FractionInsideSdf(double phi0, double phi1) const
{
    if(phi0 < 0 && phi1 < 0)
    {
        return 1;
    }
    else if(phi0 < 0 && phi1 > 0)
    {
        return phi0 / (phi0 + phi1);
    }
    else if(phi0 > 0 && phi1 < 0)
    {
        return phi1 / (phi0 + phi1);
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

