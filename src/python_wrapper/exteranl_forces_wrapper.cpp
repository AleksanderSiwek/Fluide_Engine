#include "external_forces_wrapper.hpp"

#include <memory>

#include "../forces/directional_field.hpp"
#include "../forces/point_field.hpp"
#include "../forces/swirl_field.hpp"
#include "../forces/volume_field.hpp"

namespace py = pybind11;

void addExternalForces(py::module& m)
{
    py::class_<ExternalForce, std::shared_ptr<ExternalForce>>(m, "ExternalForce");

    py::class_<DirectionalField, std::shared_ptr<DirectionalField>, ExternalForce>(m, "DirectionalField")
        .def(py::init<Vector3<double>>());

    py::class_<PointField, std::shared_ptr<PointField>, ExternalForce>(m, "PointField")
        .def(py::init<Vector3<double>, double, double>());

    py::class_<SwirlField, std::shared_ptr<SwirlField>, ExternalForce>(m, "SwirlField")
        .def(py::init<Vector3<double>, Vector3<double>, double, double>());

    py::class_<VolumeField, std::shared_ptr<VolumeField>, ExternalForce>(m, "VolumeField")
        .def(py::init<TriangleMesh, Vector3<size_t>, BoundingBox3D, double>());
}