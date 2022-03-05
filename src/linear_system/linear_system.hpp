#ifndef _LINEAR_SYSTEM_HPP
#define _LINEAR_SYSTEM_HPP

#include "../common/array3.hpp"


// TO DO: Create compressed version

struct LinearSystemMatrixRow
{
    double center = 0.0;
    double right = 0.0;
    double up = 0.0;
    double front = 0.0;
};

typedef Array3<struct LinearSystemMatrixRow> SystemMatrix;

class LinearSystem
{
    public: 
        LinearSystem();
        ~LinearSystem();

        void Resize(const Vector3<size_t>& size);
        void Clear();

        SystemMatrix A;
        Array3<double> b;
        Array3<double> x;
};

#endif // _LINEAR_SYSTEM_HPP