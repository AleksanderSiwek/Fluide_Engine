#ifndef _SERIALIZABLE_HPP
#define _SERIALIZABLE_HPP

#include <vector>


class Serializable
{
    public:
        Serializable()
        {
            
        }

        virtual ~Serializable() 
        {

        }

        virtual std::vector<double> Serialize() const = 0;
};


#endif _SERIALIZABLE_HPP