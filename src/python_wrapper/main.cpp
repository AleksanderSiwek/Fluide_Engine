#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "frame_wrapper.hpp"
#include "vector3_wrapper.hpp"
#include "bounding_box_3d_wrapper.hpp"
#include "triangle_mesh_wrapper.hpp"
#include "external_forces_wrapper.hpp"
#include "obj_manager_wrapper.hpp"
#include "pic_simulator_wrapper.hpp"

namespace py = pybind11;


PYBIND11_MODULE(PyFluidEngine, m) 
{
    addFrame(m);
    addVector3(m);
    addBoundingBox3D(m);
    addTriangleMesh(m);
    addExternalFrorces(m);
    addOBJManager(m);
    addPICSimulator(m);
}

