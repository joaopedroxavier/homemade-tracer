#pragma once

#include <Scatterable.cuh>

namespace Material {

class Metallic : public Scatterable {
public:
    __host__ __device__ Metallic() : albedo() {}
    __host__ __device__ Metallic(const Geometry::Vector3 a) : albedo(a) {}

    __device__ virtual bool scatter(
            const Geometry::Ray& r,
            const Geometry::HitRecord& record,
            curandState* rng,
            Material::ScatterRecord& scatterRecord) const;
private:
    Geometry::Vector3 albedo;
};

} // namespace Material