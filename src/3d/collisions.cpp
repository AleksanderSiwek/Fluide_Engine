#include "collisions.hpp"


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