#ifndef _PIC_SIMULATOR_WRAPPER_HPP
#define _PIC_SIMULATOR_WRAPPER_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addSimulators(py::module& m);

#endif // _PIC_SIMULATOR_WRAPPER_HPP