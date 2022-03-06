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

void LinearSystem::Build(const FaceCenteredGrid3D& input, const Array3<size_t>& markers)
{
    Vector3<size_t> size = input.GetSize();
    Resize(size);
    Vector3<double> invH = 1.0 / input.GetGridSpacing();
    Vector3<double> invHSqr = invH * invH;

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0 ; k < size.z; k++)
            {
                auto& row = A(i, j, k);

                row.center = row.right =  row.up = row.front = 0.0;
                b(i, j, k) = 0.0;

                if(markers(i, j, k) == 1)
                {
                    b(i, j, k) = input.DivergenceAtCallCenter(i, j, k);

                    if(i + 1 < size.x && markers(i + 1, j, k) != 2)
                    {
                        row.center += invHSqr.x;
                        if(markers(i + 1, j, k) == 1)
                            row.right -= invHSqr.x;
                    }
                    if(i > 0 && markers(i - 1, j, k) != 2)
                        row.center += invHSqr.x;

                    if(j + 1 < size.y && markers(i , j + 1, k) != 2)
                    {
                        row.center += invHSqr.y;
                        if(markers(i, j + 1, k) == 1)
                            row.up -= invHSqr.y;
                    }
                    if(j > 0 && markers(i, j - 1, k) != 2)
                        row.center += invHSqr.y;

                    if(k + 1 < size.z && markers(i , j, k + 1) != 2)
                    {
                        row.center += invHSqr.z;
                        if(markers(i, j, k + 1) == 1)
                            row.front -= invHSqr.z;
                    }
                    if(k > 0 && markers(i, j, k - 1) != 2)
                        row.center += invHSqr.z;
                }
                else
                {
                    row.center = 1.0;
                }
            }
        }
    }
}