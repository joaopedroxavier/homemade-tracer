#include "Metallic.h"

namespace Material {

Geometry::Vector3 reflect(
        const Geometry::Vector3& v, 
        const Geometry::Vector3& n) {
    return v - 2 * (v * n) * n;
}

ScatterRecord Metallic::scatter(
        const Geometry::Ray& r,
        const Geometry::HitRecord& record) const {
    Geometry::Ray scatteredRay = Geometry::Ray(
        record.hitPoint, 
        reflect(r.direction(), record.surfaceNormal)
    );

    return {
        scatteredRay.direction() * record.surfaceNormal > 0,
        albedo,
        scatteredRay
    };
}
    
} // namespace Material