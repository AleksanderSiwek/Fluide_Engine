#include "fluid_markers.hpp"

FluidMarkers::FluidMarkers(Vector3<size_t> size, enum FluidMarker initialMarker)
    : Array3<enum FluidMarker>(size, initialMarker)
{

}

FluidMarkers::~FluidMarkers()
{

}

void FluidMarkers::BuildFluidMarkers()
{

}