#include "flip_simulator.hpp"


FLIPSimulator::FLIPSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain)
    : PICSimulator(gridSize, domain)
{

}

FLIPSimulator::~FLIPSimulator()
{

}