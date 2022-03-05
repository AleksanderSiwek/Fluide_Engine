#include "linear_system.hpp"

LinearSystem::LinearSystem()
{
    
}

LinearSystem::~LinearSystem()
{

}

void LinearSystem::Resize(const Vector3<size_t>& size)
{
    A.Resize(size);
    b.Resize(size);
    x.Resize(size);
}

void LinearSystem::Clear()
{
    A.Clear();
    b.Clear();
    x.Clear();
}