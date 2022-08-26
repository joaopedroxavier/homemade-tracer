#pragma once

#include <Scatterable.cuh>

namespace Material {

class Glass : public Scatterable {
public:
    __host__ __device__ Glass() : refractionIndex() {}
    __host__ __device__ Glass(float r) : refractionIndex(r) {}

    __device__ virtual bool scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& hitRecord,
        curandState* rng, 
        Material::ScatterRecord& scatterRecord) const;
private:
    float refractionIndex;
};

} // namespace Material