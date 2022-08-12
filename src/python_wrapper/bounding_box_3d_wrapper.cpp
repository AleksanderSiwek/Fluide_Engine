
#include "bounding_box_3d_wrapper.hpp"

#include "../3d/bounding_box_3d.hpp"

namespace py = pybind11;

void addBoundingBox3D(py::module& m)
{
    py::class_<BoundingBox3D>(m, "BoundingBox3D")
        .def(py::init<Vector3<double>, Vector3<double>>())
        .def("GetOrigin", &BoundingBox3D::GetOrigin)
        .def("GetSize", &BoundingBox3D::GetSize);
}
