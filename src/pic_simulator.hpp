#ifndef _PIC_SIMULATOR_HPP
#define _PIC_SIMULATOR_HPP

#include <memory>

#include "physics_animation.hpp"
#include "fluid3.hpp"
#include "particle_systems/particle_system.hpp"
#include "grid_systems/gird_3d_system.hpp"
#include "fluid_solvers/diffusion_solver.hpp"
#include "fluid_solvers/pressure_solver.hpp"
#include "file_systems/file_system.hpp"


class PICSimulator : public PhysicsAnimation
{
    public:
        PICSimulator();
        
        ~PICSimulator();

        void SetViscosity(double viscosity);
        void SetDensity(double density);
        void SetDiffusionSolver(std::shared_ptr<DiffusionSolver> diffusionSolver);
        void SetPressureSolver(std::shared_ptr<PressureSolver> pressureSolver);

        double GetViscosity() const;
        double GetDensity() const;

        // TO DO: do something with that
        double Cfl(double timeIntervalInSceonds) const;
        double MaxCfl(double timeIntervalInSceonds) const;
        void SetMaxClf(double maxClf);

    protected:
        void OnInitialize() override;
        void OnAdvanceTimeStep(double timeIntervalInSeconds) override;
        virtual void OnBeginAdvanceTimeStep(double timeIntervalInSeconds);
        virtual void OnEndAdvanceTimeStep(double timeIntervalInSeconds);

        virtual void ComputeExternalForces(double timeIntervalInSeconds);
        virtual void ComputeDiffusion(double timeIntervalInSeconds);
        virtual void ComputePressure(double timeIntervalInSeconds);
        virtual void ComputeAdvection(double timeIntervalInSeconds);

        void ApplyBoundryCondition();

    private:
        Fluid3 fluid;

        std::shared_ptr<DiffusionSolver> _diffusionSolver;
        std::shared_ptr<PressureSolver> _pressureSolver;

        FileSystem _fileSystem;

        void BeginAdvanceTimeStep(double tmeIntervalInSecons);
        void EndAdvanceTimeStep(double tmeIntervalInSecons);
};

#endif // _PIC_SIMULATOR_HPP