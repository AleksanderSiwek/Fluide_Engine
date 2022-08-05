#ifndef _APIC_SIMULATOR_HPP
#define _APIC_SIMULATOR_HPP

#include "pic_simulator.hpp"


class APICSimulator : public PICSimulator
{
    public:
        APICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);

        ~APICSimulator();

    private:
};

#endif // _APIC_SIMULATOR_HPP