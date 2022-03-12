#ifndef _FLUID_MARKERS_HPP
#define _FLUID_MARKERS_HPP

#include "../common/array3.hpp"


enum FluidMarker
{
    AIR_MARK = 0,
    FLUID_MARK = 1,
    BOUNDRY_MARK = 2,
};


class FluidMarkers : public Array3<enum FluidMarker>
{
    public:
        FluidMarkers(Vector3<size_t> size = 1, enum FluidMarker initialMarker = AIR_MARK);

        ~FluidMarkers();

        void BuildFluidMarkers();

    private:
};


#endif // _FLUID_MARKERS_HPP