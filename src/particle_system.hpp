#ifndef PARTICLE_SYSTEH_HPP
#define PARTICLE_SYSTEH_HPP

#include <vector>
#include <map>
#include <string>
#include "vector3.hpp"

// TO DO search neighbours 
class ParticleSystem
{
    public:
        ParticleSystem(size_t numberOfParticles = 0);

        ~ParticleSystem();

        void Resize(size_t numberOfParticles);
        void AddPartices(size_t numberOfParticles, std::vector<Vector3<double>> positions);

        void AddScalarValue(std::string name, double initialValue=0);
        void AddVectorValue(std::string name, Vector3<double> initialValue=(0, 0, 0));

        void SetScalarValue(size_t idx, std::vector<double> values);
        void SetScalarValue(std::string name, std::vector<double> values);
        void SetVectorValue(size_t idx, std::vector<Vector3<double>> values);
        void SetVectorValue(std::string name, std::vector<Vector3<double>> values);
        void SetMass(double mass);
        void SetRadius(double radius);

        size_t GetParticleNumber() const;
        std::vector<double> GetScalarValue(size_t idx) const;
        std::vector<double> GetScalarValue(const std::string& name) const;
        std::vector<Vector3<double>> GetVectorValue(size_t idx) const;
        std::vector<Vector3<double>> GetVectorValue(const std::string& name) const;
        size_t GetScalarIdxByName(std::string name) const;
        size_t GetVectorIdxByName(std::string name) const;
        size_t GetScalarDataMaxIdx() const;
        size_t GetVectorDataMaxIdx() const;
        std::vector<std::vector<double>> GetRawScalarData() const;
        std::vector<std::vector<Vector3<double>>> GetRawVectorData() const;
        double GetMass() const;
        double GetRadius() const;

    private:
        size_t _numberOfParticles;
        double _mass = 0;
        double _radius = 0;

        std::map<std::string, size_t> _scalarDataDict;
        std::map<std::string, size_t> _vectorDataDict;
        std::vector<std::vector<double>> _scalarData;
        std::vector<std::vector<Vector3<double>>> _vectorData;

};

#endif // PARTICLE_SYSTEH_HPP