#include "triangle_mesh_wrapper.hpp"

#include "../3d/triangle_mesh.hpp"

namespace py = pybind11;


void addTriangleMesh(py::module& m)
{
    py::class_<TriangleMesh>(m, "TriangleMesh")
        .def(py::init<>())
        .def("Clear", &TriangleMesh::Clear);

}