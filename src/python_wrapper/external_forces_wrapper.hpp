#ifndef _EXTERNAL_FORCES_WRAPPER_HPP
#define _EXTERNAL_FORCES_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addExternalFrorces(py::module& m);

#endif // _DIRECTIONAL_FIELD_PTR_WRAPPER_HPP