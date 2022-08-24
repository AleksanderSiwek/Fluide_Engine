#ifndef _SCALAR_GRID_3D_WRAPPER
#define _SCALAR_GRID_3D_WRAPPER

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

void addScalarGrid(py::module& m);

#endif // _SCALAR_GRID_3D_WRAPPER