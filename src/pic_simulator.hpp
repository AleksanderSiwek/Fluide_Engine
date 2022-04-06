#ifndef _PIC_SIMULATOR_HPP
#define _PIC_SIMULATOR_HPP

#include <memory>

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
#include "file_systems/file_system.hpp"
#include "forces/external_force.hpp"
#include "3d/bounding_box_3d.hpp"


class PICSimulator : public PhysicsAnimation
{
    public:
        PICSimulator(const Vector3<size_t>& resolution, const Vector3<double> gridSpacing, const Vector3<double> gridOrigin);
        
        ~PICSimulator();

        void SetViscosity(double viscosity);
        void SetDensity(double density);
        void SetDiffusionSolver(std::shared_ptr<DiffusionSolver> diffusionSolver);
        void SetPressureSolver(std::shared_ptr<PressureSolver> pressureSolver);

        double GetViscosity() const;
        double GetDensity() const;
        Vector3<double> GetOrigin() const;
        Vector3<size_t> GetResolution() const;
        Vector3<double> GetGridSpacing() const;

        double Cfl(double timeIntervalInSceonds) const;
        double MaxCfl() const;
        void SetMaxClf(double maxClf);

    protected:
        void OnInitialize() override;
        void OnAdvanceTimeStep(double timeIntervalInSeconds) override;
        virtual void OnBeginAdvanceTimeStep(double timeIntervalInSeconds);
        virtual void OnEndAdvanceTimeStep(double timeIntervalInSeconds);

        virtual void ComputeExternalForces(double timeIntervalInSeconds);
        virtual void ComputeDiffusion(double timeIntervalInSeconds);
        virtual void ComputePressure(double timeIntervalInSeconds);
        virtual void MoveParticles(double timeIntervalInSeconds);

        void ApplyBoundryCondition();

    protected:
        unsigned int NumberOfSubTimeSteps(double tmeIntervalInSecons) const override;

    private:
        Fluid3 _fluid;

        std::shared_ptr<DiffusionSolver> _diffusionSolver;
        std::shared_ptr<PressureSolver> _pressureSolver;
        std::shared_ptr<BoundryConditionSolver> _boundryConditionSolver;
        std::vector<std::shared_ptr<VectorField3>> _externalForces;

        FileSystem _fileSystem;
        double _maxClf;

        void BeginAdvanceTimeStep(double tmeIntervalInSecons);
        void EndAdvanceTimeStep(double tmeIntervalInSecons);
};

#endif // _PIC_SIMULATOR_HPP