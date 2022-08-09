#pragma once

#include <Ray.cuh>
#include <HitRecord.cuh>

namespace Material {

struct ScatterRecord {
    Geometry::Vector3 attenuation;
    Geometry::Ray scatteredRay;
};

class Scatterable {
public:
    __device__ virtual bool scatter(
            const Geometry::Ray& r,
            const Geometry::HitRecord& hitRecord,
            curandState *rng,
            Material::ScatterRecord& scatterRecord) const = 0;
};

} // namespace Material
