#ifndef FLIP_SIMULATOR_HPP
#define FLIP_SIMULATOR_HPP

#include "pic_simulator.hpp"


class FLIPSimulator : public PICSimulator
{
    public:
        FLIPSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);

        ~FLIPSimulator();

        void SetBlendingFactor(double factor);

        double GetBlendingFactor() const;

    protected:
        void TransferParticles2Grid() override;
        void TransferGrid2Particles() override;

    private:
        double _blendingFactor;
        FaceCenteredGrid3D _velocityResiduals;

};

#endif // FLIP_SIMULATOR_HPP