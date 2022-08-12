#include "simulators_wrapper.hpp"

#include "../pic_simulator.hpp"
#include "../apic_simulator.hpp"
#include "../flip_simulator.hpp"

namespace py = pybind11;

// class PyHybridSimulator : public HybridSimulator 
// {
//  public:
//     using HybridSimulator::HybridSimulator;

//     double Cfl(double timeIntervalInSeconds) const override {
//         PYBIND11_OVERLOAD_PURE(double, HybridSimulator, Cfl,
//                                timeIntervalInSeconds);
//     }

//     void GetSurface(TriangleMesh& mesh) override {
//         PYBIND11_OVERLOAD(void, HybridSimulator, GetSurface, mesh);
//     }

//     void InitializeFromTriangleMesh(const TriangleMesh& mesh) override {
//         PYBIND11_OVERLOAD(void, HybridSimulator, InitializeFromTriangleMesh, mesh);
//     } 

//     void AddExternalForce(const std::shared_ptr<ExternalForce> newForce) override {
//         PYBIND11_OVERLOAD(void, HybridSimulator, AddExternalForce, newForce);
//     } 

//     void AddCollider(std::shared_ptr<Collider> collider) override {
//         PYBIND11_OVERLOAD(void, HybridSimulator, AddCollider, collider);
//     }                        
// };

void addSimulators(py::module& m)
{
    py::class_<HybridSimulator, std::shared_ptr<HybridSimulator>>(m, "HybridSimulator")
        .def("MaxCfl", &HybridSimulator::MaxCfl)
        .def("GetViscosity", &HybridSimulator::GetViscosity)
        .def("GetViscosity", &HybridSimulator::GetViscosity)
        .def("GetParticlesPerBlock", &HybridSimulator::GetParticlesPerBlock)
        .def("GetOrigin", &HybridSimulator::GetOrigin)
        .def("GetResolution", &HybridSimulator::GetResolution)
        .def("GetOrigin", &HybridSimulator::GetOrigin)
        .def("GetGridSpacing", &HybridSimulator::GetGridSpacing)
        .def("GetNumberOfParticles", &HybridSimulator::GetNumberOfParticles);

    py::class_<PICSimulator, std::shared_ptr<PICSimulator>, HybridSimulator>(m, "PICSimulator")
        .def(py::init<Vector3<size_t>, BoundingBox3D>())
        .def("AdvanceSingleFrame", &PICSimulator::AdvanceSingleFrame)
        .def("InitializeFromTriangleMesh", &PICSimulator::InitializeFromTriangleMesh)
        .def("AddExternalForce", &PICSimulator::AddExternalForce)
        .def("AddCollider", &PICSimulator::AddCollider)
        .def("SetCurrentFrame", &PICSimulator::SetCurrentFrame)
        .def("SetViscosity", &PICSimulator::SetViscosity)
        .def("SetMaxClf", &PICSimulator::SetMaxClf)
        .def("GetSurface", &PICSimulator::GetSurface);

    py::class_<APICSimulator, std::shared_ptr<APICSimulator>, PICSimulator>(m, "APICSimulator")
        .def(py::init<Vector3<size_t>, BoundingBox3D>())
        .def("AdvanceSingleFrame", &APICSimulator::AdvanceSingleFrame)
        .def("InitializeFromTriangleMesh", &APICSimulator::InitializeFromTriangleMesh)
        .def("AddExternalForce", &APICSimulator::AddExternalForce)
        .def("AddCollider", &APICSimulator::AddCollider)
        .def("SetCurrentFrame", &APICSimulator::SetCurrentFrame)
        .def("SetViscosity", &APICSimulator::SetViscosity)
        .def("SetMaxClf", &APICSimulator::SetMaxClf)
        .def("GetSurface", &APICSimulator::GetSurface);


    py::class_<FLIPSimulator, std::shared_ptr<FLIPSimulator>, PICSimulator>(m, "FLIPSimulator")
        .def(py::init<Vector3<size_t>, BoundingBox3D>())
        .def("AdvanceSingleFrame", &FLIPSimulator::AdvanceSingleFrame)
        .def("InitializeFromTriangleMesh", &FLIPSimulator::InitializeFromTriangleMesh)
        .def("AddExternalForce", &FLIPSimulator::AddExternalForce)
        .def("AddCollider", &FLIPSimulator::AddCollider)
        .def("SetCurrentFrame", &FLIPSimulator::SetCurrentFrame)
        .def("SetViscosity", &FLIPSimulator::SetViscosity)
        .def("SetMaxClf", &FLIPSimulator::SetMaxClf)
        .def("GetSurface", &FLIPSimulator::GetSurface);
}