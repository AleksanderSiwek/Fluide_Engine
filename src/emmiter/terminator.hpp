#ifndef _TERMINATOR_HPP
#define _TERMINATOR_HPP

#include "../particle_systems/particle_system.hpp"


class Terminator
{
    public:
        Terminator();

        ~Terminator();

        virtual void Terminate(ParticleSystem& system, size_t idx) = 0;
};

#endif // _TERMINATOR_HPP