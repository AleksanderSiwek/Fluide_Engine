#include "particle_system.hpp"


ParticleSystem::ParticleSystem(size_t numberOfParticles) : _numberOfParticles(numberOfParticles)
{
    _vectorDataDict["position"] = 0;
    _vectorData.push_back(std::vector<Vector3<double>>(numberOfParticles, 0));
}

ParticleSystem::~ParticleSystem()
{

}

void ParticleSystem::Resize(size_t numberOfParticles)
{
    _numberOfParticles = numberOfParticles;

    for (auto& data : _scalarData) {
        data.resize(_numberOfParticles, 0.0);
    }

    for (auto& data : _vectorData) {
        data.resize(_numberOfParticles, Vector3<double>(0, 0, 0));
    }
}

void ParticleSystem::AddPartices(size_t numberOfParticles, std::vector<Vector3<double>> positions)
{
    _numberOfParticles += numberOfParticles;
    Resize(_numberOfParticles);
    for(size_t i = 0; i < numberOfParticles; i++)
    {
        _vectorData[_vectorDataDict["position"]][_numberOfParticles - numberOfParticles  + i] = positions[i];
    }
}

void ParticleSystem::AddScalarValue(std::string name, double initialValue)
{
    _scalarDataDict[name] = _scalarData.size();
    _scalarData.emplace_back(_numberOfParticles, initialValue);
}

void ParticleSystem::AddVectorValue(std::string name, Vector3<double> initialValue)
{
    _vectorDataDict[name] = _vectorData.size();
    _vectorData.emplace_back(_numberOfParticles, initialValue);
}

void ParticleSystem::SetScalarValue(size_t idx, std::vector<double> values)
{
    _scalarData[idx] = values;
}

void ParticleSystem::SetScalarValue(std::string name, std::vector<double> values)
{
    _scalarData[_scalarDataDict[name]] = values;
}

void ParticleSystem::SetVectorValue(size_t idx, std::vector<Vector3<double>> values)
{
    _vectorData[idx] = values;
}

void ParticleSystem::SetVectorValue(std::string name, std::vector<Vector3<double>> values)
{
    _vectorData[_vectorDataDict[name]] = values;
}

void ParticleSystem::SetMass(double mass)
{
    _mass = mass;
}

void ParticleSystem::SetRadius(double radius)
{
    _radius = radius;
}

size_t ParticleSystem::GetParticleNumber() const
{
    return _numberOfParticles;
}

size_t ParticleSystem::GetScalarIdxByName(std::string name) const
{
    return _scalarDataDict.at(name);
}

size_t ParticleSystem::GetVectorIdxByName(std::string name) const
{
    return _vectorDataDict.at(name);
}

std::vector<double> ParticleSystem::GetScalarValue(size_t idx) const
{
    return _scalarData[idx];
}

std::vector<double> ParticleSystem::GetScalarValue(const std::string& name) const
{
    return _scalarData[_scalarDataDict.at(name)];
}

std::vector<Vector3<double>> ParticleSystem::GetVectorValue(size_t idx) const
{
    return _vectorData[idx];
}

std::vector<Vector3<double>> ParticleSystem::GetVectorValue(const std::string& name) const
{
    return _vectorData[_vectorDataDict.at(name)];
}

size_t ParticleSystem::GetScalarDataMaxIdx() const
{
    return _scalarDataDict.size();
}

size_t ParticleSystem::GetVectorDataMaxIdx() const
{
    return _vectorDataDict.size();
}

std::vector<std::vector<double>> ParticleSystem::GetRawScalarData() const
{
    return _scalarData;
}

std::vector<std::vector<Vector3<double>>> ParticleSystem::GetRawVectorData() const
{
    return _vectorData;
}

double ParticleSystem::GetMass() const
{
    return _mass;
}

double ParticleSystem::GetRadius() const
{
    return _radius;
}
