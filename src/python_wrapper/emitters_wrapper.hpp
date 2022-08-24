#ifndef _EMITTERS_WRAPPER_HPP
#define _EMITTERS_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addEmitters(py::module& m);

#endif // _EMITTERS_WRAPPER_HPP