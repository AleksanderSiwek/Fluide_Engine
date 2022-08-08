#include "pic_simulator_wrapper.hpp"

#include "../pic_simulator.hpp"

namespace py = pybind11;

void addPICSimulator(py::module& m)
{
    py::class_<HybridSimulator>(m, "HybridSimulator");

    py::class_<PICSimulator, HybridSimulator>(m, "PICSimulator")
        .def(py::init<Vector3<size_t>, BoundingBox3D>())
        .def("AdvanceSingleFrame", &PICSimulator::AdvanceSingleFrame)
        .def("InitializeFromTriangleMesh", &PICSimulator::InitializeFromTriangleMesh)
        .def("AddExternalForce", &PICSimulator::AddExternalForce)
        .def("AddCollider", &PICSimulator::AddCollider)
        .def("SetCurrentFrame", &PICSimulator::SetCurrentFrame)
        .def("SetViscosity", &PICSimulator::SetViscosity)
        .def("SetMaxClf", &PICSimulator::SetMaxClf)
        .def("GetSurface", &PICSimulator::GetSurface);
}