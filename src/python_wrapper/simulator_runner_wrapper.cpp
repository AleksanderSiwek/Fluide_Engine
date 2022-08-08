#include "simulator_runner_wrapper.hpp"

#include "../simulation_runner.hpp"
#include "../simulation_stats.hpp"

namespace py = pybind11;

void addSimulatorRunner(py::module& m)
{
    py::class_<SimulationStats>(m, "SimulationStats")
        .def(py::init<>())
        .def("PrintStats", &SimulationStats::PrintStats)
        .def("Clear", &SimulationStats::Clear)
        .def_readwrite("simulatorType", &SimulationStats::simulatorType)
        .def_readwrite("numberOfIterations", &SimulationStats::numberOfIterations)
        .def_readwrite("gridSize", &SimulationStats::gridSize)
        .def_readwrite("numberOfParticles", &SimulationStats::numberOfParticles)
        .def_readwrite("timeIntervalInSeconds", &SimulationStats::timeIntervalInSeconds)
        .def_readwrite("simulationTimeInSeconds", &SimulationStats::simulationTimeInSeconds)
        .def_readwrite("iterationTimeInSeconds", &SimulationStats::iterationTimeInSeconds)
        .def_readwrite("cflPerIteration", &SimulationStats::cflPerIteration);

    py::class_<SimulationRunner>(m, "SimulationRunner")
        .def(py::init<>())
        .def("RunSimulation", &SimulationRunner::RunSimulation)
        .def("GetStats", &SimulationRunner::GetStats);
}