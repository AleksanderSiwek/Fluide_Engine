#ifndef _HYBRID_SIMULATOR_HPP
#define _HYBRID_SIMULATOR_HPP

#include "physics_animation.hpp"

#include <memory>
#include <algorithm>

#include "physics_animation.hpp"
#include "fluid3.hpp"
#include "common/vector_field3.hpp"
#include "particle_systems/particle_system.hpp"
#include "grid_systems/gird_3d_system.hpp"
#include "fluid_solvers/diffusion_solver.hpp"
#include "fluid_solvers/pressure_solver.hpp"
#include "fluid_solvers/boundry_condition_solver.hpp"
#include "fluid_solvers/backward_euler_diffusion_solver.hpp"
#include "fluid_solvers/single_phase_pressure_solver.hpp"
#include "fluid_solvers/blocked_boundry_condition_solver.hpp"
#include "fluid_solvers/cuda_blocked_boundry_condition_solver.hpp"
#include "file_systems/file_system.hpp"
#include "forces/external_force.hpp"
#include "3d/bounding_box_3d.hpp"
#include "3d/marching_cubes_solver.hpp"
#include "3d/collisions.hpp"
#include "3d/mesh_2_sdf.hpp"
#include "3d/collider_collection.hpp"


class HybridSimulator : public PhysicsAnimation
{
    public:
        HybridSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);

        virtual ~HybridSimulator();

        virtual void InitializeFrom3dMesh(const TriangleMesh& mesh);

        void AddExternalForce(const std::shared_ptr<ExternalForce> newForce);
        void AddCollider(std::shared_ptr<Collider> collider);

        void SetViscosity(double viscosity);
        void SetDensity(double density);
        void SetDiffusionSolver(std::shared_ptr<DiffusionSolver> diffusionSolver);
        void SetPressureSolver(std::shared_ptr<PressureSolver> pressureSolver);
        void SetColliders(std::shared_ptr<ColliderCollection> colliders);

        double GetViscosity() const;
        double GetDensity() const;
        Vector3<double> GetOrigin() const;
        Vector3<size_t> GetResolution() const;
        Vector3<double> GetGridSpacing() const;
        void GetSurface(TriangleMesh* mesh);
        const ScalarGrid3D& GetFluidSdf() const;

        virtual double Cfl(double timeIntervalInSceonds) const = 0;
        double MaxCfl() const;
        void SetMaxClf(double maxClf);

    protected:
        Fluid3 _fluid;
        BoundingBox3D _domain;

        std::shared_ptr<DiffusionSolver> _diffusionSolver;
        std::shared_ptr<PressureSolver> _pressureSolver;
        std::shared_ptr<BoundryConditionSolver> _boundryConditionSolver;
        std::shared_ptr<SurfaceTracker> _surfaceTracker;
        std::vector<std::shared_ptr<ExternalForce>> _externalForces;

        double _maxClf;
        size_t _particlesPerBlok;

        unsigned int _maxNumberOfSubSteps;
        double _cflTolerance;

        virtual void OnInitialize() override;
        virtual void OnAdvanceTimeStep(double timeIntervalInSeconds) override;
        virtual void OnBeginAdvanceTimeStep(double timeIntervalInSeconds);
        virtual void OnEndAdvanceTimeStep(double timeIntervalInSeconds);

        virtual void ComputeExternalForces(double timeIntervalInSeconds);
        virtual void ComputeDiffusion(double timeIntervalInSeconds);
        virtual void ComputePressure(double timeIntervalInSeconds);
        virtual void ComputeAdvection(double timeIntervalInSeconds);
        virtual void MoveParticles(double timeIntervalInSeconds) = 0;
        virtual void TransferParticles2Grid() = 0;
        virtual void TransferGrid2Particles() = 0;

        void ApplyBoundryCondition();

        unsigned int NumberOfSubTimeSteps(double tmeIntervalInSecons) const override;
        
        void BeginAdvanceTimeStep(double tmeIntervalInSecons);
        void EndAdvanceTimeStep(double tmeIntervalInSecons);
        virtual void BuildSignedDistanceField();
        virtual void ExtrapolateVelocityToAir() = 0;  
        virtual void ExtrapolateIntoCollider() = 0;

        virtual void InitializeParticles() = 0;
};


#endif _HYBRID_SIMULATOR_HPP