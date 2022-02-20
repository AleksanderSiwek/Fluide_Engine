#ifndef _OBJECT3_HPP
#define _OBJECT3_HPP

#include "vector3.hpp"

class Object3
{
    public:
        Object3() {}
        Object3(Vector3<double> position) : _position(position) {}

        virtual ~Object3() {}

        Vector3<double> GetPosition() const { return _position; }

        void SetPosition(Vector3<double> position) { _position = position; }

    private:
        Vector3<double> _position;
};

#endif // _OBJECT3_HPP 