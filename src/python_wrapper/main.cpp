#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "frame_wrapper.hpp"

namespace py = pybind11;


PYBIND11_MODULE(PyFluidEngine, m) 
{
    m.doc() = "Fluid simulation engine for computer graphics applications";
    addFrame(m);
}

