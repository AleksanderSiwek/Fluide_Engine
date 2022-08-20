#include "collisions.hpp"
#include <iostream>

#define MAX_DISTANCE 1.7976931348623157E+30

double Collisions::DistanceToPoint(Vector3<double> p1, Vector3<double> p2)
{
    return (p2 - p1).GetLength();
}

Vector3<double> Collisions::ClossestPointOnLine(Vector3<double> point, const Line& line)
{
    return 1;
}

double Collisions::DistanceToLine(Vector3<double> point, const Line& line)
{
    return 1;
}

Vector3<double> Collisions::ClossestPointOnTriangle(Vector3<double> point, Vector3<double> trianglePoint0, Vector3<double> trianglePoint1, Vector3<double> trianglePoint2)
{
    Vector3<double> diff = trianglePoint0 - point;
    Vector3<double> edge0 = trianglePoint1 - trianglePoint0;
    Vector3<double> edge1 = trianglePoint2 - trianglePoint0;
    double a00 = edge0.Dot(edge0);
    double a01 = edge0.Dot(edge1);
    double a11 = edge1.Dot(edge1);
    double b0 = diff.Dot(edge0);
    double b1 = diff.Dot(edge1);
    double det = std::max(a00 * a11 - a01 * a01, 0.0);
    double s = a01 * b1 - a11 * b0;
    double t = a01 * b0 - a00 * b1;

    if (s + t <= det)
    {
        if (s < 0)
        {
            if (t < 0)  // region 4
            {
                if (b0 < 0)
                {
                    t = 0;
                    if (-b0 >= a00)
                    {
                        s = 1;
                    }
                    else
                    {
                        s = -b0 / a00;
                    }
                }
                else
                {
                    s = 0;
                    if (b1 >= 0)
                    {
                        t = 0;
                    }
                    else if (-b1 >= a11)
                    {
                        t = 1;
                    }
                    else
                    {
                        t = -b1 / a11;
                    }
                }
            }
            else  // region 3
            {
                s = 0;
                if (b1 >= 0)
                {
                    t = 0;
                }
                else if (-b1 >= a11)
                {
                    t = 1;
                }
                else
                {
                    t = -b1 / a11;
                }
            }
        }
        else if (t < 0)  // region 5
        {
            t = 0;
            if (b0 >= 0)
            {
                s = 0;
            }
            else if (-b0 >= a00)
            {
                s = 1;
            }
            else
            {
                s = -b0 / a00;
            }
        }
        else  // region 0
        {
            // minimum at interior point
            s /= det;
            t /= det;
        }
    }
    else
    {
        double tmp0{}, tmp1{}, numer{}, denom{};

        if (s < 0)  // region 2
        {
            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - 2 * a01 + a11;
                if (numer >= denom)
                {
                    s = 1;
                    t = 0;
                }
                else
                {
                    s = numer / denom;
                    t = 1 - s;
                }
            }
            else
            {
                s = 0;
                if (tmp1 <= 0)
                {
                    t = 1;
                }
                else if (b1 >= 0)
                {
                    t = 0;
                }
                else
                {
                    t = -b1 / a11;
                }
            }
        }
        else if (t < 0)  // region 6
        {
            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0)
            {
                numer = tmp1 - tmp0;
                denom = a00 - 2 * a01 + a11;
                if (numer >= denom)
                {
                    t = 1;
                    s = 0;
                }
                else
                {
                    t = numer / denom;
                    s = 1 - t;
                }
            }
            else
            {
                t = 0;
                if (tmp1 <= 0)
                {
                    s = 1;
                }
                else if (b0 >= 0)
                {
                    s = 0;
                }
                else
                {
                    s = -b0 / a00;
                }
            }
        }
        else  // region 1
        {
            numer = a11 + b1 - a01 - b0;
            if (numer <= 0)
            {
                s = 0;
                t = 1;
            }
            else
            {
                denom = a00 - 2 * a01 + a11;
                if (numer >= denom)
                {
                    s = 1;
                    t = 0;
                }
                else
                {
                    s = numer / denom;
                    t = 1 - s;
                }
            }
        }
    }

    return trianglePoint0 + s * edge0 + t * edge1;
}

double Collisions::DistanceToTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3)
{
    return DistanceToPoint(point, ClossestPointOnTriangle(point, p1, p2, p3));
}

Vector3<double> Collisions::RotatePointAroundAxis(const Vector3<double>& axisOrigin, const Vector3<double>& axisDirection, const Vector3<double>& point, double angle)
{
    double angleInRadians = angle * PI / 180;
    Vector3<double> normalizedDirection = axisDirection.GetNormalized();
    double x = point.x;
    double y = point.y;
    double z = point.z;
    double u = normalizedDirection.x;
    double v = normalizedDirection.y;
    double w = normalizedDirection.z;
    double a = axisOrigin.x;
    double b = axisOrigin.y;
    double c = axisOrigin.z;
    Vector3<double> rotated(0, 0, 0);
    rotated.x = (a*(v*v + w*w) - u*(b*v + c*w - u*x - v*y - w*z))*(1 - cos(angleInRadians)) + x*cos(angleInRadians) + ((-1)*c*v + b*w - w*y + v*z)*sin(angleInRadians);
    rotated.y = (b*(u*u + w*w) - v*(a*u + c*w - u*x - v*y - w*z))*(1 - cos(angleInRadians)) + y*cos(angleInRadians) + (c*u - a*w + w*x - u*z)*sin(angleInRadians);
    rotated.z = (c*(u*u + v*v) - w*(a*u + b*v - u*x - v*y - w*z))*(1 - cos(angleInRadians)) + z*cos(angleInRadians) + ((-1)*b*u + a*v - v*x + u*y)*sin(angleInRadians);
    return rotated;
}

double Collisions::DistanceToAxis(const Vector3<double>& axisOrigin, const Vector3<double>& axisDirection, const Vector3<double>& point)
{
    Vector3<double> direction = (axisDirection - axisOrigin) / Vector3<double>(DistanceToPoint(axisDirection, axisOrigin));
    Vector3<double> v = point - axisOrigin;
    double t = v.Dot(direction);
    Vector3<double> closestPointOnAxis = axisOrigin + t * direction;
    return DistanceToPoint(closestPointOnAxis, point);
}


size_t Collisions::ClosestTriangleIdx(Vector3<double> point, const TriangleMesh& mesh)
{
    const auto& triangles = mesh.GetTriangles();
    const auto& verts = mesh.GetVerticies();
    double closestDistnace = MAX_DISTANCE;
    size_t idx = 0;
    for(size_t i = 0; i < triangles.size(); i++)
    {
        double distance = DistanceToTriangle(point, verts[triangles[i].point1Idx], verts[triangles[i].point2Idx], verts[triangles[i].point3Idx]);
        if(distance < closestDistnace)
        {
            closestDistnace = distance;
            idx = i;
        }
    }
    return idx;
}


