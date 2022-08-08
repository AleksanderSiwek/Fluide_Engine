#include "obj_manager_wrapper.hpp"

#include "../3d/obj_manager.hpp"

namespace py = pybind11;

void addOBJManager(py::module& m)
{
    py::class_<OBJManager>(m, "OBJManager")
    .def(py::init<>())
    .def("Save", &OBJManager::Save)
    .def("Load", &OBJManager::Load);
}
