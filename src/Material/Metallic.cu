#include <Metallic.cuh>

namespace Material {

__device__ bool Metallic::scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& record,
        curandState* rng,
        Material::ScatterRecord& scatterRecord) const {
    Geometry::Ray scatteredRay = Geometry::Ray(
        record.hitPoint, 
        Geometry::reflect(r.direction(), record.surfaceNormal)
    );

    scatterRecord.attenuation = albedo;
    scatterRecord.scatteredRay = scatteredRay;

    return (scatteredRay.direction() * record.surfaceNormal > 0);
}
    
} // namespace Material