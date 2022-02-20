#ifndef PIC_SIMULATOR_HPP
#define PIC_SIMULATOR_HPP

#include <memory>

#include "physics_animation.hpp"
#include "particle_systems/particle_system.hpp"
#include "grid_systems/gird_3d_system.hpp"
#include "solvers/diffusion_solver.hpp"
#include "solvers/pressure_solver.hpp"
#include "file_systems/file_system.hpp"

class PICSimulator : public PhysicsAnimation
{
    public:
        PICSimulator();
        
        ~PICSimulator();

    private:
        Grid3DSystem _gridSystem;
        ParticleSystem _particleSystem;

        std::unique_ptr<DiffusionSolver> _diffusionSolver;
        std::unique_ptr<PressureSolver> _pressureSolver;

        FileSystem _fileSystem;
};

#endif // PIC_SIMULATOR_HPP