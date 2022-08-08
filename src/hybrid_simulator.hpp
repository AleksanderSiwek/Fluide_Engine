#ifndef _HYBRID_SIMULATOR_HPP
#define _HYBRID_SIMULATOR_HPP

#include "physics_animation.hpp"
#include "fluid3.hpp"
#include "3d/bounding_box_3d.hpp"
#include "3d/triangle_mesh.hpp"
#include "3d/collider.hpp"
#include "forces/external_force.hpp"


class HybridSimulator : public PhysicsAnimation
{
    public:
        HybridSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);
        virtual ~HybridSimulator();

        virtual void OnAdvanceTimeStep(double timeIntervalInSeconds) override;
        virtual void OnInitialize() override;
        virtual unsigned int NumberOfSubTimeSteps(double tmeIntervalInSecons) const override;

        virtual double Cfl(double timeIntervalInSceonds) const = 0;
        virtual void GetSurface(TriangleMesh& mesh) = 0;
        virtual void InitializeFromTriangleMesh(const TriangleMesh& mesh) = 0;
        virtual void AddExternalForce(const std::shared_ptr<ExternalForce> newForce) = 0;
        virtual void AddCollider(std::shared_ptr<Collider> collider) = 0;

        double MaxCfl() const;
        double GetViscosity() const;
        double GetDensity() const;
        size_t GetParticlesPerBlock() const;
        Vector3<double> GetOrigin() const;
        Vector3<size_t> GetResolution() const;
        Vector3<double> GetGridSpacing() const;
        const ScalarGrid3D& GetFluidSdf() const;
        size_t GetNumberOfParticles() const;

        void SetMaxClf(double maxClf);
        void SetViscosity(double viscosity);
        void SetDensity(double density);
        void SetParticlesPerBlock(size_t particlesPerBlock);

    protected:
        Fluid3 _fluid;
        BoundingBox3D _domain;
        double _maxCfl;
        unsigned int _maxNumberOfSubSteps;
        size_t _particlesPerBlok;

        const std::string PARTICLE_POSITION_KEY = "postion";
        const std::string PARTICLE_VELOCITY_KEY = "velocity";
};


#endif // _HYBRID_SIMULATOR_HPP