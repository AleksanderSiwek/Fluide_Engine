#include "animation.hpp"

Animation::Animation()
{
    
}

Animation::~Animation()
{
    
}

void Animation::Update(const Frame& frame)
{
    OnUpdate(frame);
}

