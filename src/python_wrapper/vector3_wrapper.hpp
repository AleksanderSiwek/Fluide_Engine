#ifndef _VECTOR3_WRAPPER_HPP
#define _VECTOR3_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addVector3(py::module& m);

#endif // _VECTOR3_WRAPPER_HPP