#ifndef _CUDA_PIC_SIMULATOR_HPP
#define _CUDA_PIC_SIMULATOR_HPP

#include <memory>
#include <algorithm>

#include "physics_animation.hpp"
#include "fluid3.hpp"
#include "common/vector_field3.hpp"
#include "common/cuda_array_utils.hpp"
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


class CudaPICSimulator : public PhysicsAnimation
{
    public:
        CudaPICSimulator();
        CudaPICSimulator(const Vector3<size_t>& gridSize, const BoundingBox3D& domain);

        ~CudaPICSimulator();

        void InitializeFromTriangleMesh(const TriangleMesh& mesh);

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
        void GetSurface(TriangleMesh& mesh);
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

        // CUDA
        double* _d_xVelocity;
        double* _d_yVelocity;
        double* _d_zVelocity;
        double* _d_fluidSdf;
        int* _d_xMarkers;
        int* _d_yMarkers;
        int* _d_zMarkers;
        CUDA_Vector3* _d_particlesPosition;
        CUDA_Vector3* _d_particlesVelocity;

        double* _h_xVelocity;
        double* _h_yVelocity;
        double* _h_zVelocity;
        double* _h_fluidSdf;
        int* _h_xMarkers;
        int* _h_yMarkers;
        int* _h_zMarkers;
        CUDA_Vector3* _h_particlesPosition;
        CUDA_Vector3* _h_particlesVelocity;

        bool _wasCudaInitialized;

        const int _xThreads = 4;
        const int _yThreads = 4;
        const int _zThreads = 4;

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

        virtual void AllocateCudaMemory();
        virtual void ReAllocateCudaMemory();
        virtual void FreeCudaMemory();
        virtual void CopyFluidStateHostToDevice();
        virtual void CopyFluidStateDeviceToHost();
        void CopyParticlesHostToDevice();
        void CopyParticlesDeviceToHost();
        void CopyGridHostToDevice();
        void CopyGridDeviceToHost();

    private:

};

#endif // _CUDA_PIC_SIMULATOR_HPP