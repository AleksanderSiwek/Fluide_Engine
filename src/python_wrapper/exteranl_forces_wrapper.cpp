#include "external_forces_wrapper.hpp"

#include <memory>

#include "../forces/directional_field.hpp"
#include "../forces/point_field.hpp"

namespace py = pybind11;

void addExternalForces(py::module& m)
{
    py::class_<ExternalForce, std::shared_ptr<ExternalForce>>(m, "ExternalForce");

    py::class_<DirectionalField, std::shared_ptr<DirectionalField>, ExternalForce>(m, "DirectionalField")
        .def(py::init<Vector3<double>>());

    py::class_<PointField, std::shared_ptr<PointField>, ExternalForce>(m, "PointField")
        .def(py::init<Vector3<double>, double, double>());
}