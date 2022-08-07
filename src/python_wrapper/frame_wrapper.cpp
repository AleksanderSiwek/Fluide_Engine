#include "../common/frame.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


void addFrame(py::module& m)
{
    py::class_<Frame>(m, "Frame")
        .def("__init__", [](Frame& instance, py::args args, py::kwargs kwargs) 
        {
        });
}