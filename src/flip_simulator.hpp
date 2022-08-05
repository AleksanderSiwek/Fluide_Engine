#ifndef FLIP_SIMULATOR_HPP
#define FLIP_SIMULATOR_HPP

#include "pic_simulator.hpp"


class FLIPSimulator : public PICSimulator
{
    public:
        FLIPSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);

        ~FLIPSimulator();

    private:
};

#endif // FLIP_SIMULATOR_HPP