#ifndef _COLLIDERS_WRAPPER_HPP
#define _COLLIDERS_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addColliders(py::module& m);

#endif // _COLLIDERS_WRAPPER_HPP