#ifndef _OBJ_MANAGER_WRAPPER_HPP
#define _OBJ_MANAGER_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addOBJManager(py::module& m);

#endif // _OBJ_MANAGER_WRAPPER_HPP