#ifndef _FRAME_WRAPPER_HPP
#define _FRAME_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


void addFrame(py::module& m);

#endif // _FRAME_WRAPPER_HPP