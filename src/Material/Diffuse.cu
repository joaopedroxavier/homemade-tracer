#define _USE_MATH_DEFINES
#include <math.h>

#include <Diffuse.cuh>

namespace Material {

__device__ Geometry::Vector3 randomInUnitSphere(curandState *rng) {
    float r = curand_uniform(rng);
    float theta = 2.0f * M_PI * curand_uniform(rng);
    float phi = 2.0f * M_PI * curand_uniform(rng);

    return {
        r * cos(theta) * cos(phi),
        r * cos(theta) * sin(phi),
        r * sin(theta)
    };
}

__device__ bool Diffuse::scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& hitRecord,
        curandState* rng,
        Material::ScatterRecord& scatterRecord) const {
    Geometry::Vector3 target = hitRecord.hitPoint + hitRecord.surfaceNormal + randomInUnitSphere(rng);
    Geometry::Ray scattered = Geometry::Ray(hitRecord.hitPoint, target - hitRecord.hitPoint);

    scatterRecord.attenuation = albedo;
    scatterRecord.scatteredRay = Geometry::Ray(hitRecord.hitPoint, target - hitRecord.hitPoint);

    return true;
}

} // namespace Material