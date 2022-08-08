#ifndef _BOUNDING_BOX_3D_WRAPPER_HPP
#define _BOUNDING_BOX_3D_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addBoundingBox3D(py::module& m);

#endif // _BOUNDING_BOX_3D_WRAPPER_HPP