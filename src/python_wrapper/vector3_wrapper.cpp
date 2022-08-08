#include "vector3_wrapper.hpp"

#include "../common/vector3.hpp"

namespace py = pybind11;


void addVector3(py::module& m)
{
    py::class_<Vector3<double>>(m, "Vector3D")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vector3<double>::x)
        .def_readwrite("y", &Vector3<double>::y)
        .def_readwrite("z", &Vector3<double>::z);

    py::class_<Vector3<size_t>>(m, "Vector3S")
        .def(py::init<size_t, size_t, size_t>())
        .def_readwrite("x", &Vector3<size_t>::x)
        .def_readwrite("y", &Vector3<size_t>::y)
        .def_readwrite("z", &Vector3<size_t>::z);
}