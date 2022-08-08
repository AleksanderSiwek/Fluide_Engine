#ifndef _TRIANGLE_MESH_WRAPPER_HPP
#define _TRIANGLE_MESH_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addTriangleMesh(py::module& m);

#endif // _TRIANGLE_MESH_WRAPPER_HPP