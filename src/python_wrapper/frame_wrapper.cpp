#include "frame_wrapper.hpp"

#include "../common/frame.hpp"

namespace py = pybind11;


void addFrame(py::module& m)
{
    py::class_<Frame>(m, "Frame")
        .def(py::init<double>());
}