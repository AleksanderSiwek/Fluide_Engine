#ifndef _COLLISIONS_HPP
#define _COLLISIONS_HPP

#include <utility>

#include "triangle_3d.hpp"
#include "../common/vector3.hpp"
#include "../common/math_utils.hpp"


typedef std::pair<Vector3<double>, Vector3<double>> Line;

class Collisions
{
    public:
        static double DistanceToPoint(Vector3<double> p1, Vector3<double> p2);

        static Vector3<double> ClossestPointOnLine(Vector3<double> point, const Line& line);
        static double DistanceToLine(Vector3<double> point, const Line& line);

        static Vector3<double> ClossestPointOnTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3);
        static double DistanceToTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3);
};


#endif // _COLLISIONS_HPP