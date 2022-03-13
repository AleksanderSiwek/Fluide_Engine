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

Vector3<double> Collisions::ClossestPointOnTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3)
{
    Vector3<double> edge0 = p2 - p1;
    Vector3<double> edge1 = p3 - p1;
    Vector3<double> pv = p1 - point;

    double a = edge0.Dot(edge0);
    double b = edge0.Dot(edge1);
    double c = edge1.Dot(edge1);
    double d = edge0.Dot(pv);
    double e = edge1.Dot(pv);

    double det = a*c - b*b;
    double deps = 1e-9;
    if (std::abs(det) < deps) 
    {
        // degenerate triangle: closest point is located on a line segment
        double feps = 1e-6;
        double len1 = (p1 - p2).GetLength();
        double len2 = (p2 - p3).GetLength();
        double len3 = (p3 - p1).GetLength();
        if (len1 < feps && len2 < feps && len3 < feps) 
        {
            return p1;
        }
        Vector3<double> dp, dq;
        if (len1 < feps) 
        {
            dp = p1;
            dq = p3;
        } 
        else if (len2 < feps) 
        {
            dp = p2;
            dq = p1;
        } 
        else 
        {
            dp = p3;
            dq = p2;
        }

        double dotnum = (point - dp).Dot((dq - dp));
        double dotdem = (dq - dp).Dot((dq - dp));
        if (std::abs(dotdem) < feps) 
        {
            return p1;
        } 
        else 
        {
            double lambda = dotnum / dotdem;
            if (lambda <= 0.0f) 
            {
                return dp;
            } 
            else if (lambda >= 1.0f) 
            {
                return dq;
            } 
            else 
            {
                return dp + lambda * (dq - dp);
            }
        }
    }

    double s = b*e - c*d;
    double t = b*d - a*e;

    if (s + t < det) 
    {
        if (s < 0.0) 
        {
            if (t < 0.0) 
            {
                if (d < 0.0) 
                {
                    s = Clamp<double>(-d / a, 0.0, 1.0);
                    t = 0.0;
                }
                else 
                {
                    s = 0.0;
                    t = Clamp<double>(-e / c, 0.0, 1.0);
                }
            } 
            else 
            {
                s = 0.0;
                t = Clamp<double>(-e / c, 0.0, 1.0);
            }
        } 
        else if (t < 0.0) 
        {
            s = Clamp<double>(-d / a, 0.0, 1.0);
            t = 0.0;
        } 
        else 
        {
            double invDet = 1.0 / det;
            s *= invDet;
            t *= invDet;
        }
    } 
    else 
    {
        if (s < 0.0) 
        {
            double tmp0 = b + d;
            double tmp1 = c + e;
            if (tmp1 > tmp0) 
            {
                double numer = tmp1 - tmp0;
                double denom = a - 2 * b + c;
                s = Clamp<double>(numer / denom, 0.0, 1.0);
                t = 1 - s;
            } 
            else 
            {
                t = Clamp<double>(-e / c, 0.0, 1.0);
                s = 0.0;
            }
        } 
        else if (t < 0.0) 
        {
            if (a + d > b + e) 
            {
                double numer = c + e - b - d;
                double denom = a - 2 * b + c;
                s = Clamp<double>(numer / denom, 0.0, 1.0);
                t = 1 - s;
            } 
            else 
            {
                s = Clamp<double>(-e / c, 0.0, 1.0);
                t = 0.0;
            }
        } 
        else 
        {
            double numer = c + e - b - d;
            double denom = a - 2 * b + c;
            s = Clamp<double>(numer / denom, 0.0, 1.0);
            t = 1.0 - s;
        }
    }

    return p1 + s * edge0 + t * edge1;
}

double Collisions::DistanceToTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3)
{
    return DistanceToPoint(point, ClossestPointOnTriangle(point, p1, p2, p3));
}