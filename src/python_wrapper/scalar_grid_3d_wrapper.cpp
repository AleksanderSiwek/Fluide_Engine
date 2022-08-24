#include "scalar_grid_3d_wrapper.hpp"

#include "../3d/scalar_grid3d.hpp"
#include "../common/grid3d.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

void addScalarGrid(py::module& m)
{
    py::class_<ScalarGrid3D>(m, "ScalarGrid3D")
        .def(py::init<const Vector3<size_t>&, const double&, Vector3<double>, Vector3<double>>())
        .def("GetElement", py::overload_cast<size_t, size_t, size_t>(&ScalarGrid3D::GetElement))
        .def("SetElement", &ScalarGrid3D::SetElement);

}