#ifndef _PIC_SIMULATOR_HPP
#define _PIC_SIMULATOR_HPP

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


class PICSimulator : public PhysicsAnimation
{
    public:
        PICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);
        
        ~PICSimulator();

        void InitializeFrom3dMesh(const TriangleMesh& mesh);

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

        double Cfl(double timeIntervalInSceonds) const;
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

        FileSystem _fileSystem;
        double _maxClf;
        size_t _particlesPerBlok;

        const std::string PARTICLE_POSITION_KEY = "postion";
        const std::string PARTICLE_VELOCITY_KEY = "velocity";

        unsigned int _maxNumberOfSubSteps;
        double _cflTolerance;

        void OnInitialize() override;
        void OnAdvanceTimeStep(double timeIntervalInSeconds) override;
        virtual void OnBeginAdvanceTimeStep(double timeIntervalInSeconds);
        virtual void OnEndAdvanceTimeStep(double timeIntervalInSeconds);

        virtual void ComputeExternalForces(double timeIntervalInSeconds);
        virtual void ComputeDiffusion(double timeIntervalInSeconds);
        virtual void ComputePressure(double timeIntervalInSeconds);
        virtual void ComputeAdvection(double timeIntervalInSeconds);
        virtual void MoveParticles(double timeIntervalInSeconds);

        void ApplyBoundryCondition();

        unsigned int NumberOfSubTimeSteps(double tmeIntervalInSecons) const override;
        virtual void TransferParticles2Grid();
        virtual void TransferGrid2Particles();

        void BeginAdvanceTimeStep(double tmeIntervalInSecons);
        void EndAdvanceTimeStep(double tmeIntervalInSecons);
        void BuildSignedDistanceField();
        void ExtrapolateVelocityToAir();  
        void ExtrapolateIntoCollider();

        void InitializeParticles();
};

#endif // _PIC_SIMULATOR_HPP