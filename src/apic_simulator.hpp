#ifndef _APIC_SIMULATOR_HPP
#define _APIC_SIMULATOR_HPP

#include "pic_simulator.hpp"

#include <vector>


class APICSimulator : public PICSimulator
{
    public:
        APICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);

        ~APICSimulator();

    protected:
        void TransferParticles2Grid() override;
        void TransferGrid2Particles() override;

    private:
        std::vector<Vector3<double>> _cX;
        std::vector<Vector3<double>> _cY;
        std::vector<Vector3<double>> _cZ;
};

#endif // _APIC_SIMULATOR_HPP