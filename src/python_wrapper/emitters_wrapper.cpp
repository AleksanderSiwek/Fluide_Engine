#include "emitters_wrapper.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../common/vector3.hpp"
#include "../emmiter/emmiter.hpp"
#include "../emmiter/volume_emmiter.hpp"
#include "../3d/triangle_mesh.hpp"

namespace py = pybind11;


void addEmitters(py::module& m)
{    
    py::class_<Emitter, std::shared_ptr<Emitter>>(m, "Emitter");

    py::class_<VolumeEmitter, std::shared_ptr<VolumeEmitter>, Emitter>(m, "VolumeEmitter")
        .def(py::init<const TriangleMesh&, Vector3<size_t>, BoundingBox3D, const size_t&, const Vector3<double>&, const Vector3<double>&>())
        .def("InitializeFromTriangleMesh", &VolumeEmitter::InitializeFromTriangleMesh)
        .def("InitializeFromSdf", &VolumeEmitter::InitializeFromSdf);
}