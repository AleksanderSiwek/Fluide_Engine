#ifndef PARTICLE_SYSTEH_HPP
#define PARTICLE_SYSTEH_HPP

#include <vector>
#include <map>
#include <string>
#include <memory>

#include "./hash_grid_particle_searcher.hpp"
#include "../common/vector3.hpp"


class ParticleSystem
{
    public:
        ParticleSystem(size_t numberOfParticles = 0);

        ~ParticleSystem();

        void Resize(size_t numberOfParticles);
        void BuildSearcher(std::string name, double radious);

        void AddParticles(size_t numberOfParticles, const std::vector<Vector3<double>>& vectorVal, const std::string& name);
        void AddScalarValue(std::string name, double initialValue=0);
        void AddVectorValue(std::string name, Vector3<double> initialValue=Vector3<double>(0, 0, 0));

        void ForEachParticle(std::function<void(size_t)>& functor);
        void ForEachParticle(const std::function<void(size_t)>& functor) const;
        void ParallelForEachParticle(std::function<void(size_t)>& functor);
        void ParallelForEachParticle(const std::function<void(size_t)>& functor) const;

        void SetScalarValues(size_t idx, std::vector<double> values);
        void SetScalarValues(std::string name, std::vector<double> values);
        void SetVectorValues(size_t idx, std::vector<Vector3<double>> values);
        void SetVectorValues(std::string name, std::vector<Vector3<double>> values);
        void SetScalarValue(size_t idx, size_t particleIdx, double values);
        void SetScalarValue(std::string name, size_t particleIdx, double value);
        void SetVectorValue(size_t idx, size_t particleIdx, Vector3<double> value);
        void SetVectorValue(std::string name,size_t particleIdx, Vector3<double> value);
        void SetMass(double mass);
        void SetRadius(double radius);
        void SetParticleSystemSearcher(std::shared_ptr<ParticleSystemSearcher> particleSearcher);

        ParticleSystemSearcher* GetSearcher();
        size_t GetParticleNumber() const;
        std::vector<double>& GetScalarValues(size_t idx);
        std::vector<double>& GetScalarValues(const std::string& name);
        std::vector<Vector3<double>>& GetVectorValues(size_t idx);
        std::vector<Vector3<double>>& GetVectorValues(const std::string& name);
        const std::vector<double>& GetScalarValues(size_t idx) const;
        const std::vector<double>& GetScalarValues(const std::string& name) const;
        const std::vector<Vector3<double>>& GetVectorValues(size_t idx) const;
        const std::vector<Vector3<double>>& GetVectorValues(const std::string& name) const;
        const std::vector<double>* GetScalarValuesPtr(size_t idx);
        std::vector<double>* GetScalarValuesPtr(const std::string& name);
        std::vector<Vector3<double>>* GetVectorValuesPtr(size_t idx);
        std::vector<Vector3<double>>* GetVectorValuesPtr(const std::string& name);
        double GetScalarValue(size_t idx, size_t particleIdx) const;
        double GetScalarValue(const std::string& name, size_t particleIdx) const;
        Vector3<double> GetVectorValue(size_t idx, size_t particleIdx) const;
        Vector3<double> GetVectorValue(const std::string& name, size_t particleIdx) const;
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
        std::shared_ptr<ParticleSystemSearcher> _particleSearcher;
};

#endif // PARTICLE_SYSTEH_HPP