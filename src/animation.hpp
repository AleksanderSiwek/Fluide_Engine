#ifndef _ANIMATION_HPP
#define _ANIMATION_HPP

#include "./common/frame.hpp"

class Animation
{
    public:
        Animation();
    
        virtual ~Animation();

        void Update(const Frame& frame);

    protected:
        virtual void OnUpdate(const Frame& frame) = 0;

};

#endif // _ANIMATION_HPP