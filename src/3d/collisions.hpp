#ifndef _COLLISIONS_HPP
#define _COLLISIONS_HPP

#include <utility>

#include "triangle_mesh.hpp"
#include "../common/vector3.hpp"
#include "../common/math_utils.hpp"

#define _USE_MATH_DEFINES


typedef std::pair<Vector3<double>, Vector3<double>> Line;

class Collisions
{
    public:
        static double DistanceToPoint(Vector3<double> p1, Vector3<double> p2);

        static Vector3<double> ClossestPointOnLine(Vector3<double> point, const Line& line);
        static double DistanceToLine(Vector3<double> point, const Line& line);

        static Vector3<double> ClossestPointOnTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3);
        static double DistanceToTriangle(Vector3<double> point, Vector3<double> p1, Vector3<double> p2, Vector3<double> p3);
        static size_t ClosestTriangleIdx(Vector3<double> point, const TriangleMesh& mesh);

        static bool IsInsideTriangleMesh(const TriangleMesh& mesh, const Vector3<double>& point);

    private:
        static double ADet(const Vector3<double>& point1, const Vector3<double>& point2, const Vector3<double>& point3);
};


#endif // _COLLISIONS_HPP