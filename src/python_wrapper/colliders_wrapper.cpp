#include "colliders_wrapper.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../3d/collider.hpp"
#include "../3d/triangle_mesh_collider.hpp"

namespace py = pybind11;

void addColliders(py::module& m)
{
    py::class_<Collider, std::shared_ptr<Collider>>(m, "Collider");

    py::class_<TriangleMeshCollider, std::shared_ptr<TriangleMeshCollider>, Collider>(m, "TriangleMeshCollider")
        .def(py::init<const Vector3<size_t>&, const Vector3<double>&, const Vector3<double>&, const TriangleMesh&>());
}