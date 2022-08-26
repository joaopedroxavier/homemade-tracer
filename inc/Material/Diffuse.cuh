#pragma once

#include <Ray.cuh>
#include <Vector3.cuh>
#include <Hitable.cuh>
#include <Scatterable.cuh>

namespace Material {

class Diffuse : public Scatterable {
public:
    __host__ __device__ Diffuse() : albedo() {}
    __host__ __device__ Diffuse(Geometry::Vector3 c) : albedo(c) {}

    __device__ virtual bool scatter(
            const Geometry::Ray& r,
            const Geometry::HitRecord& hitRecord,
            curandState* rng,
            Material::ScatterRecord& scatterRecord) const;
private:
    Geometry::Vector3 albedo;
};

} // namespace Material
