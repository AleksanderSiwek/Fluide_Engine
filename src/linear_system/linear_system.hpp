#ifndef _LINEAR_SYSTEM_HPP
#define _LINEAR_SYSTEM_HPP

#include "../common/array3.hpp"
#include "../grid_systems/face_centered_grid3d.hpp"

// TO DO: Create compressed version
// TO DO: Create proper markers

struct LinearSystemMatrixRow
{
    double center = 0.0;
    double right = 0.0;
    double up = 0.0;
    double front = 0.0;
};

typedef Array3<struct LinearSystemMatrixRow> SystemMatrix;
typedef Array3<double> SystemVector;

class LinearSystem
{
    public: 
        LinearSystem();
        ~LinearSystem();

        void Resize(const Vector3<size_t>& size);
        void Clear();
        void Build(const FaceCenteredGrid3D& input, const Array3<size_t>& markers);

        SystemMatrix A;
        SystemVector b;
        SystemVector x;
};

#endif // _LINEAR_SYSTEM_HPP